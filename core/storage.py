import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from core.config import Settings
from core.models.common import ArtifactRef
from core.models.job import Job, JobStatus, StageResult


class ArtifactStore(Protocol):
    def put_json(self, job_id: str, stage_name: str, payload: dict[str, Any]) -> ArtifactRef: ...


class JobRepository(Protocol):
    def save_job(self, job: Job) -> None: ...
    def save_stage_result(self, job_id: str, result: StageResult) -> None: ...
    def get_job(self, job_id: str) -> Job | None: ...


class LocalArtifactStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def put_json(self, job_id: str, stage_name: str, payload: dict[str, Any]) -> ArtifactRef:
        path = self.root / job_id / f"{stage_name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        return ArtifactRef(uri=str(path), media_type="application/json")


class S3ArtifactStore:
    def __init__(
        self,
        endpoint_url: str,
        bucket: str,
        access_key_id: str,
        secret_access_key: str,
        region: str,
    ) -> None:
        try:
            import boto3
            from botocore.client import Config
            from botocore.exceptions import ClientError
        except ImportError as exc:
            raise RuntimeError("Install service dependencies with `pip install -e .[services]`.") from exc

        self.bucket = bucket
        self.endpoint_url = endpoint_url
        self._client_error = ClientError
        self.client = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key,
            region_name=region,
            config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
        )
        self._ensure_bucket()

    def _ensure_bucket(self) -> None:
        try:
            self.client.head_bucket(Bucket=self.bucket)
        except self._client_error as exc:
            status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
            if status not in {404, 400}:
                raise
            self.client.create_bucket(Bucket=self.bucket)

    def put_json(self, job_id: str, stage_name: str, payload: dict[str, Any]) -> ArtifactRef:
        key = f"{job_id}/{stage_name}.json"
        body = json.dumps(payload, indent=2, default=str).encode("utf-8")
        self.client.put_object(Bucket=self.bucket, Key=key, Body=body, ContentType="application/json")
        return ArtifactRef(
            uri=f"s3://{self.bucket}/{key}",
            media_type="application/json",
            metadata={"endpoint_url": self.endpoint_url},
        )


class SQLiteJobStore:
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


class PostgresJobStore:
    def __init__(self, dsn: str) -> None:
        try:
            import psycopg
        except ImportError as exc:
            raise RuntimeError("Install service dependencies with `pip install -e .[services]`.") from exc

        self.dsn = dsn
        self._psycopg = psycopg
        self._init_schema()

    def _connect(self):
        return self._psycopg.connect(self.dsn)

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    updated_at TIMESTAMPTZ NOT NULL
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
                    updated_at TIMESTAMPTZ NOT NULL,
                    PRIMARY KEY (job_id, stage_name)
                )
                """
            )

    def save_job(self, job: Job) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO jobs (job_id, status, payload, updated_at)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT(job_id) DO UPDATE SET
                    status = excluded.status,
                    payload = excluded.payload,
                    updated_at = excluded.updated_at
                """,
                (
                    job.job_id,
                    job.status.value,
                    job.model_dump_json(),
                    datetime.now(timezone.utc),
                ),
            )

    def save_stage_result(self, job_id: str, result: StageResult) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO stage_results (job_id, stage_name, status, payload, updated_at)
                VALUES (%s, %s, %s, %s, %s)
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
                    datetime.now(timezone.utc),
                ),
            )

    def get_job(self, job_id: str) -> Job | None:
        with self._connect() as conn:
            row = conn.execute("SELECT payload FROM jobs WHERE job_id = %s", (job_id,)).fetchone()
        if row is None:
            return None
        return Job.model_validate_json(row[0])


JobStore = SQLiteJobStore


def create_artifact_store(settings: Settings) -> ArtifactStore:
    if settings.artifact_store == "s3":
        return S3ArtifactStore(
            endpoint_url=settings.s3_endpoint_url,
            bucket=settings.s3_bucket,
            access_key_id=settings.s3_access_key_id,
            secret_access_key=settings.s3_secret_access_key,
            region=settings.s3_region,
        )
    return LocalArtifactStore(settings.artifact_dir)


def create_job_store(settings: Settings) -> JobRepository:
    if settings.job_store == "postgres":
        return PostgresJobStore(settings.postgres_dsn)
    return SQLiteJobStore(settings.sqlite_path)


def completed_stage(stage_name: str, artifact: ArtifactRef, adapter_name: str = "noop") -> StageResult:
    return StageResult(
        stage_name=stage_name,
        status=JobStatus.COMPLETED,
        artifact=artifact,
        adapter_name=adapter_name,
        completed_at=datetime.now(timezone.utc),
    )
