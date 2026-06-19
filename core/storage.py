import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.models.common import ArtifactRef
from core.models.job import Job, JobStatus, StageResult


class LocalArtifactStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def put_json(self, job_id: str, stage_name: str, payload: dict[str, Any]) -> ArtifactRef:
        path = self.root / job_id / f"{stage_name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        return ArtifactRef(uri=str(path), media_type="application/json")


class JobStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS stage_results (
                    job_id TEXT NOT NULL,
                    stage_name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (job_id, stage_name)
                )
                """
            )

    def save_job(self, job: Job) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO jobs (job_id, status, payload, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(job_id) DO UPDATE SET
                    status = excluded.status,
                    payload = excluded.payload,
                    updated_at = excluded.updated_at
                """,
                (
                    job.job_id,
                    job.status.value,
                    job.model_dump_json(),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

    def save_stage_result(self, job_id: str, result: StageResult) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO stage_results (job_id, stage_name, status, payload, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(job_id, stage_name) DO UPDATE SET
                    status = excluded.status,
                    payload = excluded.payload,
                    updated_at = excluded.updated_at
                """,
                (
                    job_id,
                    result.stage_name,
                    result.status.value,
                    result.model_dump_json(),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

    def get_job(self, job_id: str) -> Job | None:
        with self._connect() as conn:
            row = conn.execute("SELECT payload FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        if row is None:
            return None
        return Job.model_validate_json(row[0])


def completed_stage(stage_name: str, artifact: ArtifactRef, adapter_name: str = "noop") -> StageResult:
    return StageResult(
        stage_name=stage_name,
        status=JobStatus.COMPLETED,
        artifact=artifact,
        adapter_name=adapter_name,
        completed_at=datetime.now(timezone.utc),
    )

