from __future__ import annotations

import json
import os
import random
import socket
import sqlite3
import string
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.models.dashboard import (
    CreateDashboardJobRequest,
    DashboardApprovalKind,
    DashboardApprovalRequest,
    DashboardApprovalStatus,
    DashboardArtifact,
    DashboardArtifactKind,
    DashboardEvent,
    DashboardEventLevel,
    DashboardJobDetail,
    DashboardJobRecord,
    DashboardJobStatus,
    DashboardQueueAction,
    DashboardQueueItem,
    DashboardQueueStatus,
    WorkerHeartbeat,
    utc_now,
)


def _make_id(prefix: str) -> str:
    suffix = "".join(random.choices(string.ascii_lowercase + string.digits, k=6))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{prefix}_{stamp}_{suffix}"


def make_dashboard_job_id() -> str:
    suffix = "".join(random.choices(string.ascii_lowercase + string.digits, k=3))
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + f"-{suffix}"


def _json_dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str)


def _json_loads(value: str | None, default: Any) -> Any:
    if not value:
        return default
    return json.loads(value)


def _dt(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value)


def _dt_text(value: datetime | None) -> str | None:
    return value.isoformat() if value else None


class DashboardRepository:
    """SQLite-backed dashboard store.

    The schema is deliberately table-shaped rather than only JSON payloads so a
    future Postgres implementation can reuse the same API and indexes.
    """

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS dashboard_jobs (
                    job_id TEXT PRIMARY KEY,
                    source_url TEXT NOT NULL,
                    source_kind TEXT NOT NULL,
                    status TEXT NOT NULL,
                    phase TEXT NOT NULL,
                    target_language TEXT NOT NULL,
                    rights_cleared INTEGER NOT NULL,
                    current_stage TEXT,
                    current_shot_id TEXT,
                    approval_kind TEXT,
                    created_at TEXT NOT NULL,
                    queued_at TEXT,
                    started_at TEXT,
                    updated_at TEXT NOT NULL,
                    last_heartbeat_at TEXT,
                    completed_at TEXT,
                    terminal_error_json TEXT,
                    request_json TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS job_queue (
                    queue_id TEXT PRIMARY KEY,
                    job_id TEXT NOT NULL,
                    action TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    priority INTEGER NOT NULL DEFAULT 100,
                    created_at TEXT NOT NULL,
                    claimed_at TEXT,
                    claimed_by TEXT,
                    completed_at TEXT,
                    error_json TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_job_queue_status_created
                ON job_queue (status, priority, created_at)
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS job_events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    level TEXT NOT NULL,
                    stage_name TEXT,
                    shot_id TEXT,
                    message TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_job_events_job_event
                ON job_events (job_id, event_id)
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS artifacts (
                    artifact_id TEXT PRIMARY KEY,
                    job_id TEXT NOT NULL,
                    stage_name TEXT,
                    shot_id TEXT,
                    kind TEXT NOT NULL,
                    uri TEXT NOT NULL,
                    mime_type TEXT,
                    size_bytes INTEGER,
                    sha256 TEXT,
                    previewable INTEGER NOT NULL DEFAULT 0,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS approval_requests (
                    approval_id TEXT PRIMARY KEY,
                    job_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    status TEXT NOT NULL,
                    iteration INTEGER NOT NULL DEFAULT 1,
                    request_json TEXT NOT NULL,
                    response_json TEXT,
                    created_at TEXT NOT NULL,
                    decided_at TEXT,
                    reviewer TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS worker_heartbeats (
                    worker_id TEXT PRIMARY KEY,
                    hostname TEXT,
                    process_id INTEGER,
                    version TEXT,
                    current_job_id TEXT,
                    started_at TEXT NOT NULL,
                    last_heartbeat_at TEXT NOT NULL
                )
                """
            )

    # ------------------------------------------------------------------
    # Jobs
    # ------------------------------------------------------------------

    def create_queued_job(
        self,
        request: CreateDashboardJobRequest,
        *,
        priority: int = 100,
    ) -> tuple[DashboardJobRecord, DashboardQueueItem]:
        now = utc_now()
        job = DashboardJobRecord(
            job_id=make_dashboard_job_id(),
            source_url=request.source.url,
            source_kind=request.source.kind,
            status=DashboardJobStatus.QUEUED,
            phase=request.phase,
            target_language=request.target_language,
            rights_cleared=request.rights_cleared,
            created_at=now,
            queued_at=now,
            updated_at=now,
            request=request.model_dump(mode="json"),
        )
        queue_item = DashboardQueueItem(
            queue_id=_make_id("queue"),
            job_id=job.job_id,
            action=DashboardQueueAction.START,
            payload=request.model_dump(mode="json"),
            priority=priority,
            created_at=now,
        )
        with self._connect() as conn:
            self._insert_job(conn, job)
            self._insert_queue_item(conn, queue_item)
            event_id = self._insert_event(
                conn,
                job_id=job.job_id,
                event_type="job_queued",
                level=DashboardEventLevel.INFO,
                message="Job queued from dashboard API.",
                payload={"queue_id": queue_item.queue_id},
                created_at=now,
            )
        # Touch event construction for parity with other paths; caller does not
        # need the event here, but tests can still read it through list_events.
        assert event_id > 0
        return job, queue_item

    def get_job(self, job_id: str) -> DashboardJobRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM dashboard_jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
        return self._job_from_row(row) if row else None

    def list_jobs(self, *, limit: int = 50) -> list[DashboardJobRecord]:
        limit = max(1, min(limit, 200))
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM dashboard_jobs
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [self._job_from_row(row) for row in rows]

    def get_job_detail(self, job_id: str, *, event_limit: int = 100) -> DashboardJobDetail | None:
        job = self.get_job(job_id)
        if job is None:
            return None
        return DashboardJobDetail(
            job=job,
            queue=self.list_queue(job_id),
            events=self.list_events(job_id, limit=event_limit),
            pending_approval=self.get_pending_approval(job_id),
        )

    def update_job_status(
        self,
        job_id: str,
        status: DashboardJobStatus,
        *,
        current_stage: str | None = None,
        current_shot_id: str | None = None,
        approval_kind: str | None = None,
        terminal_error: dict[str, Any] | None = None,
        completed: bool = False,
    ) -> DashboardJobRecord:
        now = utc_now()
        completed_at = now if completed else None
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE dashboard_jobs
                SET status = ?,
                    current_stage = ?,
                    current_shot_id = ?,
                    approval_kind = ?,
                    updated_at = ?,
                    completed_at = COALESCE(?, completed_at),
                    terminal_error_json = ?
                WHERE job_id = ?
                """,
                (
                    status.value,
                    current_stage,
                    current_shot_id,
                    approval_kind,
                    _dt_text(now),
                    _dt_text(completed_at),
                    _json_dumps(terminal_error) if terminal_error else None,
                    job_id,
                ),
            )
        job = self.get_job(job_id)
        if job is None:
            raise KeyError(f"dashboard job not found: {job_id}")
        return job

    def heartbeat_job(
        self,
        job_id: str,
        *,
        current_stage: str | None = None,
        current_shot_id: str | None = None,
    ) -> None:
        now = utc_now()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE dashboard_jobs
                SET last_heartbeat_at = ?,
                    updated_at = ?,
                    current_stage = COALESCE(?, current_stage),
                    current_shot_id = COALESCE(?, current_shot_id)
                WHERE job_id = ?
                """,
                (_dt_text(now), _dt_text(now), current_stage, current_shot_id, job_id),
            )

    # ------------------------------------------------------------------
    # Queue
    # ------------------------------------------------------------------

    def enqueue_action(
        self,
        job_id: str,
        action: DashboardQueueAction,
        *,
        payload: dict[str, Any] | None = None,
        priority: int = 100,
    ) -> DashboardQueueItem:
        queue_item = DashboardQueueItem(
            queue_id=_make_id("queue"),
            job_id=job_id,
            action=action,
            payload=payload or {},
            priority=priority,
        )
        with self._connect() as conn:
            self._insert_queue_item(conn, queue_item)
        return queue_item

    def list_queue(self, job_id: str) -> list[DashboardQueueItem]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM job_queue
                WHERE job_id = ?
                ORDER BY created_at ASC
                """,
                (job_id,),
            ).fetchall()
        return [self._queue_from_row(row) for row in rows]

    def claim_next_action(self, worker_id: str) -> DashboardQueueItem | None:
        now = utc_now()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                """
                SELECT * FROM job_queue
                WHERE status = ?
                ORDER BY priority ASC, created_at ASC
                LIMIT 1
                """,
                (DashboardQueueStatus.QUEUED.value,),
            ).fetchone()
            if row is None:
                conn.execute("COMMIT")
                return None
            conn.execute(
                """
                UPDATE job_queue
                SET status = ?, claimed_at = ?, claimed_by = ?
                WHERE queue_id = ?
                """,
                (
                    DashboardQueueStatus.CLAIMED.value,
                    _dt_text(now),
                    worker_id,
                    row["queue_id"],
                ),
            )
            conn.execute("COMMIT")
        claimed = self.get_queue_item(row["queue_id"])
        return claimed

    def get_queue_item(self, queue_id: str) -> DashboardQueueItem | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM job_queue WHERE queue_id = ?",
                (queue_id,),
            ).fetchone()
        return self._queue_from_row(row) if row else None

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------

    def record_event(
        self,
        job_id: str,
        event_type: str,
        message: str,
        *,
        level: DashboardEventLevel = DashboardEventLevel.INFO,
        stage_name: str | None = None,
        shot_id: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> DashboardEvent:
        now = utc_now()
        with self._connect() as conn:
            event_id = self._insert_event(
                conn,
                job_id=job_id,
                event_type=event_type,
                level=level,
                message=message,
                stage_name=stage_name,
                shot_id=shot_id,
                payload=payload or {},
                created_at=now,
            )
        return DashboardEvent(
            event_id=event_id,
            job_id=job_id,
            event_type=event_type,
            level=level,
            stage_name=stage_name,
            shot_id=shot_id,
            message=message,
            payload=payload or {},
            created_at=now,
        )

    def list_events(
        self,
        job_id: str,
        *,
        after_event_id: int = 0,
        limit: int = 200,
    ) -> list[DashboardEvent]:
        limit = max(1, min(limit, 500))
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM job_events
                WHERE job_id = ? AND event_id > ?
                ORDER BY event_id ASC
                LIMIT ?
                """,
                (job_id, after_event_id, limit),
            ).fetchall()
        return [self._event_from_row(row) for row in rows]

    # ------------------------------------------------------------------
    # Approvals and artifacts
    # ------------------------------------------------------------------

    def create_approval_request(
        self,
        job_id: str,
        kind: DashboardApprovalKind,
        *,
        request: dict[str, Any],
        iteration: int = 1,
    ) -> DashboardApprovalRequest:
        approval = DashboardApprovalRequest(
            approval_id=_make_id("approval"),
            job_id=job_id,
            kind=kind,
            iteration=iteration,
            request=request,
        )
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO approval_requests (
                    approval_id, job_id, kind, status, iteration, request_json,
                    response_json, created_at, decided_at, reviewer
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    approval.approval_id,
                    approval.job_id,
                    approval.kind.value,
                    approval.status.value,
                    approval.iteration,
                    _json_dumps(approval.request),
                    None,
                    _dt_text(approval.created_at),
                    None,
                    None,
                ),
            )
        return approval

    def get_pending_approval(self, job_id: str) -> DashboardApprovalRequest | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM approval_requests
                WHERE job_id = ? AND status = ?
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (job_id, DashboardApprovalStatus.PENDING.value),
            ).fetchone()
        return self._approval_from_row(row) if row else None

    def record_artifact(
        self,
        job_id: str,
        kind: DashboardArtifactKind,
        uri: str,
        *,
        stage_name: str | None = None,
        shot_id: str | None = None,
        mime_type: str | None = None,
        size_bytes: int | None = None,
        sha256: str | None = None,
        previewable: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> DashboardArtifact:
        artifact = DashboardArtifact(
            artifact_id=_make_id("artifact"),
            job_id=job_id,
            stage_name=stage_name,
            shot_id=shot_id,
            kind=kind,
            uri=uri,
            mime_type=mime_type,
            size_bytes=size_bytes,
            sha256=sha256,
            previewable=previewable,
            metadata=metadata or {},
        )
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO artifacts (
                    artifact_id, job_id, stage_name, shot_id, kind, uri,
                    mime_type, size_bytes, sha256, previewable, metadata_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    artifact.artifact_id,
                    artifact.job_id,
                    artifact.stage_name,
                    artifact.shot_id,
                    artifact.kind.value,
                    artifact.uri,
                    artifact.mime_type,
                    artifact.size_bytes,
                    artifact.sha256,
                    int(artifact.previewable),
                    _json_dumps(artifact.metadata),
                    _dt_text(artifact.created_at),
                ),
            )
        return artifact

    def list_artifacts(self, job_id: str) -> list[DashboardArtifact]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM artifacts
                WHERE job_id = ?
                ORDER BY created_at ASC
                """,
                (job_id,),
            ).fetchall()
        return [self._artifact_from_row(row) for row in rows]

    # ------------------------------------------------------------------
    # Worker heartbeat
    # ------------------------------------------------------------------

    def heartbeat_worker(
        self,
        worker_id: str,
        *,
        current_job_id: str | None = None,
        version: str | None = None,
    ) -> WorkerHeartbeat:
        now = utc_now()
        heartbeat = WorkerHeartbeat(
            worker_id=worker_id,
            hostname=socket.gethostname(),
            process_id=os.getpid(),
            version=version,
            current_job_id=current_job_id,
            started_at=now,
            last_heartbeat_at=now,
        )
        with self._connect() as conn:
            existing = conn.execute(
                "SELECT started_at FROM worker_heartbeats WHERE worker_id = ?",
                (worker_id,),
            ).fetchone()
            started_at = _dt(existing["started_at"]) if existing else heartbeat.started_at
            conn.execute(
                """
                INSERT INTO worker_heartbeats (
                    worker_id, hostname, process_id, version, current_job_id,
                    started_at, last_heartbeat_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(worker_id) DO UPDATE SET
                    hostname = excluded.hostname,
                    process_id = excluded.process_id,
                    version = excluded.version,
                    current_job_id = excluded.current_job_id,
                    last_heartbeat_at = excluded.last_heartbeat_at
                """,
                (
                    heartbeat.worker_id,
                    heartbeat.hostname,
                    heartbeat.process_id,
                    heartbeat.version,
                    heartbeat.current_job_id,
                    _dt_text(started_at),
                    _dt_text(now),
                ),
            )
        heartbeat.started_at = started_at or now
        return heartbeat

    def latest_worker_heartbeat(self) -> WorkerHeartbeat | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM worker_heartbeats
                ORDER BY last_heartbeat_at DESC
                LIMIT 1
                """
            ).fetchone()
        return self._worker_from_row(row) if row else None

    def ping(self) -> None:
        with self._connect() as conn:
            conn.execute("SELECT 1").fetchone()

    # ------------------------------------------------------------------
    # Row mappers
    # ------------------------------------------------------------------

    def _insert_job(self, conn: sqlite3.Connection, job: DashboardJobRecord) -> None:
        conn.execute(
            """
            INSERT INTO dashboard_jobs (
                job_id, source_url, source_kind, status, phase, target_language,
                rights_cleared, current_stage, current_shot_id, approval_kind,
                created_at, queued_at, started_at, updated_at, last_heartbeat_at,
                completed_at, terminal_error_json, request_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                job.job_id,
                job.source_url,
                job.source_kind,
                job.status.value,
                job.phase,
                job.target_language,
                int(job.rights_cleared),
                job.current_stage,
                job.current_shot_id,
                job.approval_kind,
                _dt_text(job.created_at),
                _dt_text(job.queued_at),
                _dt_text(job.started_at),
                _dt_text(job.updated_at),
                _dt_text(job.last_heartbeat_at),
                _dt_text(job.completed_at),
                _json_dumps(job.terminal_error) if job.terminal_error else None,
                _json_dumps(job.request),
            ),
        )

    def _insert_queue_item(self, conn: sqlite3.Connection, item: DashboardQueueItem) -> None:
        conn.execute(
            """
            INSERT INTO job_queue (
                queue_id, job_id, action, payload_json, status, priority,
                created_at, claimed_at, claimed_by, completed_at, error_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                item.queue_id,
                item.job_id,
                item.action.value,
                _json_dumps(item.payload),
                item.status.value,
                item.priority,
                _dt_text(item.created_at),
                _dt_text(item.claimed_at),
                item.claimed_by,
                _dt_text(item.completed_at),
                _json_dumps(item.error) if item.error else None,
            ),
        )

    def _insert_event(
        self,
        conn: sqlite3.Connection,
        *,
        job_id: str,
        event_type: str,
        level: DashboardEventLevel,
        message: str,
        stage_name: str | None = None,
        shot_id: str | None = None,
        payload: dict[str, Any] | None = None,
        created_at: datetime | None = None,
    ) -> int:
        created = created_at or utc_now()
        cur = conn.execute(
            """
            INSERT INTO job_events (
                job_id, event_type, level, stage_name, shot_id,
                message, payload_json, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                job_id,
                event_type,
                level.value,
                stage_name,
                shot_id,
                message,
                _json_dumps(payload or {}),
                _dt_text(created),
            ),
        )
        return int(cur.lastrowid)

    def _job_from_row(self, row: sqlite3.Row) -> DashboardJobRecord:
        return DashboardJobRecord(
            job_id=row["job_id"],
            source_url=row["source_url"],
            source_kind=row["source_kind"],
            status=DashboardJobStatus(row["status"]),
            phase=row["phase"],
            target_language=row["target_language"],
            rights_cleared=bool(row["rights_cleared"]),
            current_stage=row["current_stage"],
            current_shot_id=row["current_shot_id"],
            approval_kind=row["approval_kind"],
            created_at=_dt(row["created_at"]) or utc_now(),
            queued_at=_dt(row["queued_at"]),
            started_at=_dt(row["started_at"]),
            updated_at=_dt(row["updated_at"]) or utc_now(),
            last_heartbeat_at=_dt(row["last_heartbeat_at"]),
            completed_at=_dt(row["completed_at"]),
            terminal_error=_json_loads(row["terminal_error_json"], None),
            request=_json_loads(row["request_json"], {}),
        )

    def _queue_from_row(self, row: sqlite3.Row) -> DashboardQueueItem:
        return DashboardQueueItem(
            queue_id=row["queue_id"],
            job_id=row["job_id"],
            action=DashboardQueueAction(row["action"]),
            payload=_json_loads(row["payload_json"], {}),
            status=DashboardQueueStatus(row["status"]),
            priority=row["priority"],
            created_at=_dt(row["created_at"]) or utc_now(),
            claimed_at=_dt(row["claimed_at"]),
            claimed_by=row["claimed_by"],
            completed_at=_dt(row["completed_at"]),
            error=_json_loads(row["error_json"], None),
        )

    def _event_from_row(self, row: sqlite3.Row) -> DashboardEvent:
        return DashboardEvent(
            event_id=row["event_id"],
            job_id=row["job_id"],
            event_type=row["event_type"],
            level=DashboardEventLevel(row["level"]),
            stage_name=row["stage_name"],
            shot_id=row["shot_id"],
            message=row["message"],
            payload=_json_loads(row["payload_json"], {}),
            created_at=_dt(row["created_at"]) or utc_now(),
        )

    def _artifact_from_row(self, row: sqlite3.Row) -> DashboardArtifact:
        return DashboardArtifact(
            artifact_id=row["artifact_id"],
            job_id=row["job_id"],
            stage_name=row["stage_name"],
            shot_id=row["shot_id"],
            kind=DashboardArtifactKind(row["kind"]),
            uri=row["uri"],
            mime_type=row["mime_type"],
            size_bytes=row["size_bytes"],
            sha256=row["sha256"],
            previewable=bool(row["previewable"]),
            metadata=_json_loads(row["metadata_json"], {}),
            created_at=_dt(row["created_at"]) or utc_now(),
        )

    def _approval_from_row(self, row: sqlite3.Row) -> DashboardApprovalRequest:
        return DashboardApprovalRequest(
            approval_id=row["approval_id"],
            job_id=row["job_id"],
            kind=DashboardApprovalKind(row["kind"]),
            status=DashboardApprovalStatus(row["status"]),
            iteration=row["iteration"],
            request=_json_loads(row["request_json"], {}),
            response=_json_loads(row["response_json"], None),
            created_at=_dt(row["created_at"]) or utc_now(),
            decided_at=_dt(row["decided_at"]),
            reviewer=row["reviewer"],
        )

    def _worker_from_row(self, row: sqlite3.Row) -> WorkerHeartbeat:
        return WorkerHeartbeat(
            worker_id=row["worker_id"],
            hostname=row["hostname"],
            process_id=row["process_id"],
            version=row["version"],
            current_job_id=row["current_job_id"],
            started_at=_dt(row["started_at"]) or utc_now(),
            last_heartbeat_at=_dt(row["last_heartbeat_at"]) or utc_now(),
        )
