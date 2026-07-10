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
import shlex
import signal
import socket
import traceback as _traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from core.config import AppConfig, load_app_config
from core.gpu_sequencer import stop_fish_s2_process
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
        self._reclaim_orphaned_jobs()

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

    def _reclaim_orphaned_jobs(self) -> None:
        """Requeue jobs left claimed by a worker that died mid-run.

        Runs once at startup. Without this, a job whose worker was killed
        (crash, OOM, `restart_dashboard.sh`) sits at status=running forever —
        claim_next_action only ever looks at QUEUED rows, so no future worker
        would otherwise pick it back up. The pipeline's own resume logic
        (RunOptions.resume, keyed on prior artifacts) picks up where the dead
        worker left off, so this is a plain requeue, not a restart-from-scratch.
        """
        reclaimed = self.repo.reclaim_stale_claims(stale_after_seconds=self.STALL_THRESHOLD)
        for job_id, previous_worker_id in reclaimed:
            logger.warning(
                "Reclaimed job %s — previous worker %s stopped heartbeating",
                job_id, previous_worker_id,
            )
            self.repo.record_event(
                job_id,
                "job_reclaimed",
                f"Requeued after worker {previous_worker_id} stopped responding "
                "(crashed or was restarted mid-job). Resuming from last completed stage.",
                level=DashboardEventLevel.WARNING,
            )

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

        try:
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
        finally:
            # Every outcome (completed/failed/cancelled), unconditionally: Fish
            # S2's own CUDA allocator retains memory across synthesis calls
            # that POST /unload can't reclaim (~63GB resident observed vs
            # ~20GB fresh-process baseline). Killing the whole process is the
            # only reliable reset; the next job that needs it respawns it on
            # demand (core.gpu_sequencer.ensure_fish_s2_process_running).
            await stop_fish_s2_process()

    async def _execute_pipeline(self, action: DashboardQueueItem) -> None:
        """Dispatch to the right workflow function based on job phase."""
        from core.models.dashboard import DashboardQueueAction as QA

        req = CreateDashboardJobRequest(**action.payload)
        job_id = action.job_id

        if req.phase == "noop":
            await self._run_noop(req, job_id)
        elif req.phase == "lora_train":
            await self._run_lora_training(req, job_id)
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

    async def _run_lora_training(
        self, req: CreateDashboardJobRequest, job_id: str
    ) -> None:
        from services.dashboard_api import _find_lora_config_path

        training = req.lora_training
        if training is None:
            raise ValueError("lora_training payload is required")

        job_config = self._config_for_job(req)
        member = next(
            (m for m in job_config.cast.members if m.id == training.cast_member_id),
            None,
        )
        if member is None:
            raise ValueError(
                f"Unknown cast member '{training.cast_member_id}' for cast '{job_config.cast.id}'"
            )

        config_path = _find_lora_config_path(member.id)
        try:
            config_arg = str(config_path.resolve().relative_to(Path.cwd().resolve()))
        except ValueError:
            config_arg = str(config_path.resolve())

        for image_path in training.image_paths:
            if not Path(image_path).is_file():
                raise FileNotFoundError(f"Training image not found: {image_path}")

        trigger = str(member.lora_ref).replace("\\", "/").rstrip("/").split("/")[-1]
        output_name = f"{job_config.cast.id}_{trigger}"
        vae_path = "/workspace/ComfyUI/models/diffusion_models/ae.safetensors"
        text_encoder_path = "/workspace/FLUX2-text-encoder/text_encoder/model-00001-of-00010.safetensors"

        self.repo.update_job_status(
            job_id,
            DashboardJobStatus.RUNNING,
            current_stage="lora_train",
        )

        if not os.getenv("VIDEO_ME_LORA_TRAIN_CMD"):
            # flux_2_train_network.py reads only from the dataset's latent/text-encoder
            # cache directory — it never reads images/captions directly. Skipping these
            # two steps leaves the cache empty and training fails immediately with
            # "No training items found in the dataset."
            self.repo.record_event(
                job_id,
                "phase_started",
                f"Caching latents for {job_config.cast.id}/{member.id}.",
                stage_name="lora_cache_latents",
            )
            await self._run_lora_subprocess(
                job_id,
                "lora_cache_latents",
                [
                    "/workspace/.venv_musubi/bin/python3",
                    "/workspace/musubi-tuner/src/musubi_tuner/flux_2_cache_latents.py",
                    "--model_version",
                    "dev",
                    "--dataset_config",
                    config_arg,
                    "--vae",
                    vae_path,
                ],
            )

            self.repo.record_event(
                job_id,
                "phase_started",
                f"Caching text encoder outputs for {job_config.cast.id}/{member.id}.",
                stage_name="lora_cache_text_encoder",
            )
            await self._run_lora_subprocess(
                job_id,
                "lora_cache_text_encoder",
                [
                    "/workspace/.venv_musubi/bin/python3",
                    "/workspace/musubi-tuner/src/musubi_tuner/flux_2_cache_text_encoder_outputs.py",
                    "--model_version",
                    "dev",
                    "--dataset_config",
                    config_arg,
                    "--text_encoder",
                    text_encoder_path,
                ],
            )

        if os.getenv("VIDEO_ME_LORA_TRAIN_CMD"):
            template = os.environ["VIDEO_ME_LORA_TRAIN_CMD"]
            template = template.format(
                config_path=config_arg,
                dataset_config=config_arg,
                output_name=output_name,
                cast_id=job_config.cast.id,
                member_id=member.id,
            )
            cmd = shlex.split(template)
            if "{config_path}" not in os.environ["VIDEO_ME_LORA_TRAIN_CMD"] and "{dataset_config}" not in os.environ["VIDEO_ME_LORA_TRAIN_CMD"]:
                cmd.extend(["--dataset_config", config_arg])
        else:
            cmd = [
                "/workspace/.venv_musubi/bin/accelerate",
                "launch",
                "/workspace/musubi-tuner/src/musubi_tuner/flux_2_train_network.py",
                "--model_version",
                "dev",
                "--dit",
                "/workspace/ComfyUI/models/diffusion_models/flux2-dev.safetensors",
                "--vae",
                vae_path,
                "--text_encoder",
                text_encoder_path,
                "--dataset_config",
                config_arg,
                "--flash_attn",
                "--mixed_precision",
                "bf16",
                "--fp8_base",
                "--fp8_scaled",
                "--gradient_checkpointing",
                "--optimizer_type",
                "adamw",
                "--learning_rate",
                "1e-4",
                "--network_module",
                "networks.lora_flux_2",
                "--network_dim",
                "32",
                "--network_alpha",
                "16",
                "--max_train_epochs",
                "25",
                "--save_every_n_epochs",
                "5",
                "--seed",
                "42",
                "--output_dir",
                "loras/",
                "--output_name",
                output_name,
            ]

        self.repo.record_event(
            job_id,
            "phase_started",
            f"Training LoRA for {job_config.cast.id}/{member.id} with {len(training.image_paths)} image(s).",
            stage_name="lora_train",
            payload={"config_path": config_arg, "command": cmd},
        )
        await self._run_lora_subprocess(job_id, "lora_train", cmd)

        self.repo.update_job_status(job_id, DashboardJobStatus.COMPLETED, completed=True)
        self.repo.append_completed_phase(job_id, "lora_train")
        self.repo.record_event(
            job_id,
            "phase_completed",
            f"LoRA training completed for {job_config.cast.id}/{member.id}.",
            stage_name="lora_train",
        )

    async def _run_lora_subprocess(
        self, job_id: str, stage_name: str, cmd: list[str]
    ) -> None:
        """Run one LoRA pipeline step (cache latents/text-encoder or train), streaming
        its output to job_events, and raise if it exits non-zero."""
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=Path.cwd(),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        assert proc.stdout is not None
        line_count = 0
        async for raw_line in proc.stdout:
            line = raw_line.decode(errors="replace").rstrip()
            if not line:
                continue
            line_count += 1
            level = (
                DashboardEventLevel.WARNING
                if "warning" in line.lower()
                else DashboardEventLevel.INFO
            )
            if line_count <= 40 or line_count % 25 == 0:
                self.repo.record_event(
                    job_id,
                    "lora_train_log",
                    line[-1000:],
                    level=level,
                    stage_name=stage_name,
                )

        returncode = await proc.wait()
        if returncode != 0:
            raise RuntimeError(f"{stage_name} failed with exit code {returncode}")

    def _make_stage_hook(self, job_id: str):
        """Return a callback that writes per-stage events to the dashboard repo.

        ``shot_id`` and ``message`` are optional keyword-only extras: callers in
        the shot loop (core/workflow.py's _generate_shot_video) pass them to
        surface which shot is in progress (current_shot_id) and a human-readable
        per-shot status line, instead of the generic "<stage>: stage started"
        text. Existing (stage_name, event_type) callers are unaffected.
        """
        def hook(
            stage_name: str,
            event_type: str,
            *,
            shot_id: str | None = None,
            message: str | None = None,
        ) -> None:
            if event_type == "stage_started":
                self.repo.update_job_status(
                    job_id,
                    DashboardJobStatus.RUNNING,
                    current_stage=stage_name,
                    current_shot_id=shot_id,
                )
            level = (
                DashboardEventLevel.ERROR
                if event_type == "stage_failed"
                else DashboardEventLevel.INFO
            )
            self.repo.record_event(
                job_id,
                event_type,
                message or f"{stage_name}: {event_type.replace('_', ' ')}",
                level=level,
                stage_name=stage_name,
                shot_id=shot_id,
            )
        return hook

    def _config_for_job(self, req: CreateDashboardJobRequest) -> AppConfig:
        """Apply the job's per-request overrides (model/adapter choices) on top
        of the worker's base config. DashboardJobOverrides' field names are a
        deliberate 1:1 match with Settings' fields, so this is a direct copy.
        Per-job cast: if req.cast_ref differs from the default, load that cast YAML.
        """
        config = self.config

        if req.cast_ref and req.cast_ref != config.cast.id:
            from pathlib import Path as _Path
            from core.config import load_yaml_model
            from core.models.profile import Cast

            cast_path = _Path(f"config/casts/{req.cast_ref}.yaml")
            if not cast_path.exists():
                raise ValueError(f"Unknown cast: {req.cast_ref} (no file at {cast_path})")
            new_cast = load_yaml_model(cast_path, Cast)
            feedback_dir = _Path(f"assets/{new_cast.id}")
            new_settings = config.settings.model_copy(update={"feedback_log_dir": feedback_dir})
            config = config.model_copy(update={"cast": new_cast, "settings": new_settings})

        overrides = req.overrides
        updates = {
            k: v for k, v in overrides.model_dump().items() if v is not None
        }
        if updates:
            new_settings = config.settings.model_copy(update=updates)
            config = config.model_copy(update={"settings": new_settings})
        return config

    async def _run_pipeline(self, req: CreateDashboardJobRequest, job_id: str) -> None:
        from core.storage import create_job_store
        from core.workflow import RunOptions, run_pipeline_job

        job_config = self._config_for_job(req)

        # Build dashboard approval adapters so approval gates use the repo
        # instead of blocking flag-file servers.
        approval_overrides = self._make_approval_overrides(
            req, job_id, settings=job_config.settings
        )

        # Story input modes: pre-seed fetch_media + transcribe artifacts from the
        # pasted story so the pipeline (run with resume=True) skips yt-dlp/Whisper.
        is_story = req.source.kind in ("story", "story_images")
        user_images: dict[str, str] | None = None
        if is_story:
            user_images = await self._seed_story_job(req, job_id)

        phase = req.phase if req.phase != "all" else "all"
        # script_plan/render/assemble resume from artifacts saved by the previous
        # phase; story jobs always resume so the seeded artifacts are picked up.
        # A prior core Job row means this is a retry/continuation of an earlier
        # attempt — resume from its cached per-stage artifacts regardless of
        # phase, instead of rerunning the whole pipeline from fetch_media.
        had_prior_run = create_job_store(job_config.settings).get_job(job_id) is not None
        resume = is_story or had_prior_run or phase in ("script_plan", "render", "assemble")
        options = RunOptions(
            phase=phase,
            resume=resume,
            render_mode=req.render_mode,
            user_images=user_images,
            stage_hook=self._make_stage_hook(job_id),
            error_hook=self._make_error_hook(job_id),
        )

        # resume_job_id restores an existing core Job row (prior phase / retry).
        # A fresh story job has no core row yet — pass job_id only, so a new row
        # is created while the seeded artifacts are still found under it.
        core_job_exists = resume and had_prior_run

        self.repo.record_event(
            job_id, "phase_started", f"Starting pipeline phase={phase}.", stage_name=phase
        )

        job = await run_pipeline_job(
            source_url=req.source.url,
            rights_cleared=req.rights_cleared,
            app_config=job_config,
            options=options,
            job_id=job_id,
            resume_job_id=job_id if core_job_exists else None,
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
            if phase == "all":
                # End-to-end run covers every macro phase — record them all so
                # completed_phases stays truthful for the stepper/artifact views.
                for sub_phase in ("transcribe", "script_plan", "render", "assemble"):
                    self.repo.append_completed_phase(job_id, sub_phase)
            self.repo.append_completed_phase(job_id, phase)
            self.repo.record_event(
                job_id, "phase_completed", f"Phase '{phase}' done.", stage_name=phase
            )

    # ------------------------------------------------------------------
    # Story input seeding
    # ------------------------------------------------------------------

    async def _seed_story_job(
        self, req: CreateDashboardJobRequest, job_id: str
    ) -> dict[str, str] | None:
        """Pre-seed fetch_media + transcribe artifacts from the pasted story.

        Idempotent: skipped when the transcribe artifact already exists (retry,
        or second language pass of target_language="both"). The fetch_media stub
        is written FIRST — the guard checks transcribe, so a mid-seed crash can
        never leave transcribe present without fetch_media (which would send the
        story:// URL into yt-dlp on retry).

        Returns the member_id → copied-image-path map for story_images jobs.
        """
        import shutil

        from adapters.story_ingest.parser import parse_structured_story
        from core.models.capabilities import FetchMediaResult
        from core.storage import create_artifact_store

        s = self.config.settings
        store = create_artifact_store(s)
        work_dir = Path(s.data_dir) / "jobs" / job_id

        if store.has(job_id, "transcribe"):
            self.repo.record_event(
                job_id, "story_seeded", "Story artifacts already present — seeding skipped."
            )
        else:
            language = "hi" if req.target_language == "hi" else "en"
            transcript = parse_structured_story(req.story_text or "", language=language)
            if transcript is not None:
                seg_source = "structured timeline"
            else:
                from adapters.story_ingest.llm_adapter import LlmStorySegmentAdapter

                segmenter = LlmStorySegmentAdapter(base_url=s.llm_base_url, model=s.llm_model)
                transcript = await segmenter.segment(req.story_text or "", language=language)
                seg_source = "LLM segmentation"

            duration = transcript.segments[-1].end if transcript.segments else 0.0
            fetch_stub = FetchMediaResult(
                video_uri=f"story://{job_id}",
                audio_uri=f"story://{job_id}",
                duration_sec=duration,
                source_url=req.source.url,
            )
            store.put_json(job_id, "fetch_media", fetch_stub.model_dump())
            store.put_json(job_id, "transcribe", transcript.model_dump())
            self.repo.record_event(
                job_id,
                "story_seeded",
                f"Story converted to {len(transcript.segments)} timed segments "
                f"via {seg_source}; fetch/transcribe stages will be skipped.",
            )

        if req.source.kind != "story_images":
            return None

        # Copy reference images under data_dir so the /img route can serve them
        # in the approval grid, and so the job work dir is self-contained.
        image_dir = work_dir / "user_images"
        image_dir.mkdir(parents=True, exist_ok=True)
        user_images: dict[str, str] = {}
        for member_id, src in req.character_images.items():
            src_path = Path(src)
            if not src_path.is_file():
                raise FileNotFoundError(f"Character image for '{member_id}' not found: {src}")
            dest = image_dir / f"{member_id}{src_path.suffix.lower()}"
            if src_path.resolve() != dest.resolve():
                shutil.copyfile(src_path, dest)
            user_images[member_id] = str(dest)
        self.repo.record_event(
            job_id,
            "story_images_staged",
            f"Reference images staged for: {', '.join(sorted(user_images))}.",
        )
        return user_images

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

        job_settings = self._config_for_job(req).settings
        if bool(getattr(job_settings, "auto_approve_transcript", False)):
            # Operator disabled this gate at job creation (unattended run).
            self.repo.update_job_status(job_id, DashboardJobStatus.COMPLETED, completed=True)
            self.repo.append_completed_phase(job_id, "transcribe")
            self.repo.record_event(
                job_id,
                "approval_granted",
                "Transcript auto-approved (review gate disabled for this job). "
                "Phase 'transcribe' done.",
                stage_name="transcribe",
            )
            logger.info("Job %s: transcript auto-approved (gate disabled)", job_id)
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
            current = self.repo.get_approval(approval_id)
            if current is None:
                continue
            if current.status == DashboardApprovalStatus.APPROVED:
                return (True, "")
            if current.status == DashboardApprovalStatus.REJECTED:
                notes = (current.response or {}).get("notes", "") if current.response else ""
                return (False, notes)
        return None

    def _load_transcript_artifact(self, job_id: str) -> dict | None:
        from core.storage import create_artifact_store

        try:
            return create_artifact_store(self.config.settings).get_json(job_id, "transcribe")
        except Exception as exc:
            logger.warning("Job %s: failed to read transcript artifact: %s", job_id, exc)
            return None

    def _save_transcript_artifact(self, job_id: str, transcript: dict) -> None:
        from core.storage import create_artifact_store

        create_artifact_store(self.config.settings).put_json(job_id, "transcribe", transcript)

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
        self, req: CreateDashboardJobRequest, job_id: str, settings: Any = None
    ) -> dict[str, Any]:
        """Return approval adapter overrides that use the dashboard repo.

        auto_approve_plan/auto_approve_images come from the job's overrides
        (merged onto Settings by _config_for_job) — checking them here matters
        because these dashboard adapters REPLACE the workflow-built ones, which
        are the only place the settings flags are otherwise honoured.
        """
        from adapters.approval.dashboard_approval_adapter import DashboardPlanApprovalAdapter
        from adapters.approval.dashboard_image_approval_adapter import (
            DashboardImageApprovalAdapter,
        )

        auto_plan = bool(getattr(settings, "auto_approve_plan", False))
        auto_images = bool(getattr(settings, "auto_approve_images", False))
        return {
            "approval": DashboardPlanApprovalAdapter(
                self.repo, job_id, auto_approve=auto_plan
            ),
            "image_approval": DashboardImageApprovalAdapter(
                self.repo, job_id, auto_approve=auto_images
            ),
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
