"""Opaque, durable media assets for dashboard jobs.

The dashboard API owns ingestion/decoding and uses this module for the parts
that must remain consistent across upload, URL, and allowlisted-server-file
entry points:

* clients receive random ``asset_id`` values rather than filesystem paths;
* staged assets expire unless a job atomically claims them;
* ownership, media kind, and claim scope are checked on every resolution; and
* every stored path is re-validated beneath the server-managed asset root.

The store deliberately shares the dashboard SQLite database but does not
depend on ``DashboardRepository``. This keeps asset claims available before a
job is queued and lets API integration release a claim if queue creation fails.
"""

from __future__ import annotations

import hashlib
import json
import re
import secrets
import sqlite3
from collections.abc import Iterable, Mapping
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from core.models.dashboard import (
    DashboardAssetKind,
    DashboardAssetRecord,
    DashboardAssetStatus,
    WAN_ANIMATE_MAX_DRIVER_RANGE_SEC,
    WanAnimateJobOptions,
    utc_now,
)


_ASSET_ID_RE = re.compile(r"^ast_[A-Za-z0-9_-]{20,64}$")
_SAFE_SUFFIX_RE = re.compile(r"^\.[A-Za-z0-9]{1,10}$")
_VIDEO_SUFFIXES = frozenset({".mp4", ".mov", ".webm", ".mkv", ".m4v", ".avi"})
_IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".webp", ".avif"})


class DashboardAssetError(RuntimeError):
    """Base class for errors that API routes can map to stable HTTP codes."""


class DashboardAssetNotFoundError(DashboardAssetError):
    pass


class DashboardAssetAccessError(DashboardAssetError):
    pass


class DashboardAssetStateError(DashboardAssetError):
    pass


class DashboardAssetKindError(DashboardAssetError):
    pass


class DashboardAssetPathError(DashboardAssetError):
    pass


class DashboardAssetMetadataError(DashboardAssetError):
    pass


class DashboardAssetQuotaError(DashboardAssetError):
    pass


def make_dashboard_asset_id() -> str:
    """Return an opaque, URL-safe identifier with 192 bits of randomness."""

    return f"ast_{secrets.token_urlsafe(24)}"


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Hash a file without loading it into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def collect_animate_asset_requirements(
    options: WanAnimateJobOptions,
) -> dict[str, DashboardAssetKind]:
    """Collect every opaque asset referenced by a direct Animate request.

    A dictionary is used so one asset cannot be silently requested as two
    different media kinds. Pydantic already rejects duplicates within each
    complete-look reference list; this function also de-duplicates across fields.
    """

    requirements: dict[str, DashboardAssetKind] = {
        options.driver.asset_id: DashboardAssetKind.VIDEO,
    }

    def add(asset_id: str, kind: DashboardAssetKind) -> None:
        previous = requirements.get(asset_id)
        if previous is not None and previous != kind:
            raise DashboardAssetKindError(
                f"asset {asset_id} is referenced as both {previous.value} and {kind.value}"
            )
        requirements[asset_id] = kind

    character = options.character
    if character.exact_image_asset_id:
        add(character.exact_image_asset_id, DashboardAssetKind.IMAGE)
    if character.wardrobe:
        for asset_id in character.wardrobe.garment_asset_ids:
            add(asset_id, DashboardAssetKind.IMAGE)
        for asset_id in character.wardrobe.accessory_asset_ids:
            add(asset_id, DashboardAssetKind.IMAGE)
    return requirements


class DashboardAssetStore:
    """SQLite-backed staged/claimed asset registry.

    Args:
        db_path: Usually ``DashboardRepository.db_path``.
        storage_root: Private server-managed directory containing normalized
            uploads. Paths outside this root can never become asset records.
        allowed_server_roots: Directories exposed by the server-file picker.
            A selected file must still be copied/normalized into
            ``storage_root`` before ``create_staged`` is called.
        default_ttl: Lifetime of an unclaimed upload.
    """

    def __init__(
        self,
        db_path: Path,
        storage_root: Path,
        allowed_server_roots: Iterable[Path] = (),
        *,
        default_ttl: timedelta = timedelta(hours=24),
        max_total_bytes: int = 50 * 1024 * 1024 * 1024,
    ) -> None:
        if default_ttl.total_seconds() <= 0:
            raise ValueError("default_ttl must be positive")
        if max_total_bytes <= 0:
            raise ValueError("max_total_bytes must be positive")
        self.db_path = Path(db_path)
        self.storage_root = Path(storage_root).expanduser().resolve()
        self.allowed_server_roots = tuple(
            Path(root).expanduser().resolve() for root in allowed_server_roots
        )
        self.default_ttl = default_ttl
        self.max_total_bytes = int(max_total_bytes)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.storage_root.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS dashboard_assets (
                    asset_id TEXT PRIMARY KEY,
                    owner_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    status TEXT NOT NULL,
                    original_name TEXT NOT NULL,
                    mime_type TEXT NOT NULL,
                    sha256 TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    storage_path TEXT NOT NULL UNIQUE,
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    claimed_job_id TEXT,
                    claimed_at TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_dashboard_assets_owner_status
                ON dashboard_assets (owner_id, status, expires_at)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_dashboard_assets_claimed_job
                ON dashboard_assets (claimed_job_id)
                """
            )

    # ------------------------------------------------------------------
    # Storage paths
    # ------------------------------------------------------------------

    def allocate_path(
        self,
        kind: DashboardAssetKind | str,
        *,
        suffix: str,
        asset_id: str | None = None,
    ) -> tuple[str, Path]:
        """Allocate a collision-resistant destination for a streamed upload.

        This does not create a database row. The caller streams into the path,
        validates/normalizes the media, and then calls ``create_staged``. On a
        failed upload the caller can safely unlink the returned path.
        """

        kind = DashboardAssetKind(kind)
        asset_id = asset_id or make_dashboard_asset_id()
        self._validate_asset_id(asset_id)
        suffix = suffix.lower()
        if not _SAFE_SUFFIX_RE.fullmatch(suffix):
            raise DashboardAssetPathError("asset suffix must be a short alphanumeric extension")
        destination = self.storage_root / kind.value / asset_id[:8] / f"{asset_id}{suffix}"
        destination.parent.mkdir(parents=True, exist_ok=True)
        return asset_id, self._storage_path(destination, must_exist=False)

    def validate_server_path(
        self,
        candidate: str | Path,
        *,
        expected_kind: DashboardAssetKind | str | None = None,
    ) -> Path:
        """Resolve a server-file selection under an explicitly allowed root.

        Absolute paths and root-relative picker values are supported. Symlink
        escapes fail because ``resolve`` runs before the containment check.
        Extension checks are defense-in-depth only; API ingestion must still
        inspect the file signature and decode the media.
        """

        if not self.allowed_server_roots:
            raise DashboardAssetPathError("server-file selection is disabled")
        raw = Path(candidate).expanduser()
        possible = [raw] if raw.is_absolute() else [root / raw for root in self.allowed_server_roots]
        resolved: Path | None = None
        for path in possible:
            try:
                current = path.resolve(strict=True)
            except (FileNotFoundError, OSError):
                continue
            if not current.is_file():
                continue
            if any(self._is_within(current, root) for root in self.allowed_server_roots):
                resolved = current
                break
        if resolved is None:
            raise DashboardAssetPathError("server file is missing or outside allowed roots")
        if expected_kind is not None:
            kind = DashboardAssetKind(expected_kind)
            allowed = _VIDEO_SUFFIXES if kind == DashboardAssetKind.VIDEO else _IMAGE_SUFFIXES
            if resolved.suffix.lower() not in allowed:
                raise DashboardAssetKindError(
                    f"server file extension is not accepted for {kind.value} assets"
                )
        return resolved

    # ------------------------------------------------------------------
    # Records and metadata
    # ------------------------------------------------------------------

    def create_staged(
        self,
        *,
        owner_id: str,
        kind: DashboardAssetKind | str,
        original_name: str,
        mime_type: str,
        storage_path: Path,
        sha256: str | None = None,
        size_bytes: int | None = None,
        metadata: Mapping[str, Any] | None = None,
        asset_id: str | None = None,
        expires_at: datetime | None = None,
        now: datetime | None = None,
    ) -> DashboardAssetRecord:
        """Register a fully streamed and validated file as a staged asset."""

        owner_id = owner_id.strip()
        if not owner_id:
            raise ValueError("owner_id is required")
        kind = DashboardAssetKind(kind)
        asset_id = asset_id or make_dashboard_asset_id()
        self._validate_asset_id(asset_id)
        path = self._storage_path(storage_path, must_exist=True)
        if not path.is_file():
            raise DashboardAssetPathError("asset storage path must be a regular file")
        normalized_name = self._safe_original_name(original_name)
        normalized_mime = mime_type.strip().lower()
        if not normalized_mime:
            raise ValueError("mime_type is required")
        actual_size = path.stat().st_size
        if size_bytes is not None and size_bytes != actual_size:
            raise DashboardAssetMetadataError("size_bytes does not match the stored file")
        content_sha = (sha256 or sha256_file(path)).lower()
        if not re.fullmatch(r"[0-9a-f]{64}", content_sha):
            raise DashboardAssetMetadataError("sha256 must contain 64 lowercase hex characters")
        now = self._utc(now or utc_now())
        expires_at = self._utc(expires_at or (now + self.default_ttl))
        if expires_at <= now:
            raise ValueError("expires_at must be in the future")
        record = DashboardAssetRecord(
            asset_id=asset_id,
            owner_id=owner_id,
            kind=kind,
            status=DashboardAssetStatus.STAGED,
            original_name=normalized_name,
            mime_type=normalized_mime,
            sha256=content_sha,
            size_bytes=actual_size,
            metadata=dict(metadata or {}),
            storage_path=str(path),
            created_at=now,
            expires_at=expires_at,
        )
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                active_bytes = int(
                    conn.execute(
                        """
                        SELECT COALESCE(SUM(size_bytes), 0) FROM dashboard_assets
                        WHERE status IN (?, ?)
                        """,
                        (
                            DashboardAssetStatus.STAGED.value,
                            DashboardAssetStatus.CLAIMED.value,
                        ),
                    ).fetchone()[0]
                )
                if active_bytes + actual_size > self.max_total_bytes:
                    raise DashboardAssetQuotaError(
                        "Dashboard input-asset quota exceeded; delete completed jobs or "
                        "staged inputs before uploading more media"
                    )
                conn.execute(
                    """
                    INSERT INTO dashboard_assets (
                        asset_id, owner_id, kind, status, original_name,
                        mime_type, sha256, size_bytes, metadata_json,
                        storage_path, created_at, expires_at,
                        claimed_job_id, claimed_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL)
                    """,
                    (
                        record.asset_id,
                        record.owner_id,
                        record.kind.value,
                        record.status.value,
                        record.original_name,
                        record.mime_type,
                        record.sha256,
                        record.size_bytes,
                        json.dumps(record.metadata, separators=(",", ":"), sort_keys=True),
                        record.storage_path,
                        self._dt(record.created_at),
                        self._dt(record.expires_at),
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise DashboardAssetStateError("asset ID or storage path already exists") from exc
        return record

    def get(self, asset_id: str) -> DashboardAssetRecord | None:
        self._validate_asset_id(asset_id)
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM dashboard_assets WHERE asset_id = ?", (asset_id,)
            ).fetchone()
        return self._from_row(row) if row else None

    def update_metadata(
        self,
        asset_id: str,
        *,
        owner_id: str,
        metadata: Mapping[str, Any] | None = None,
        mime_type: str | None = None,
        merge: bool = True,
        now: datetime | None = None,
    ) -> DashboardAssetRecord:
        """Save normalized probe/decode metadata while an asset is staged."""

        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT * FROM dashboard_assets WHERE asset_id = ?", (asset_id,)
            ).fetchone()
            record = self._require_row(row, asset_id)
            self._require_owner(record, owner_id)
            if record.status != DashboardAssetStatus.STAGED:
                raise DashboardAssetStateError("only staged asset metadata may be updated")
            if self._utc(now or utc_now()) >= self._utc(record.expires_at):
                raise DashboardAssetStateError("dashboard asset has expired")
            current = dict(record.metadata) if merge else {}
            current.update(dict(metadata or {}))
            updated_mime = (mime_type or record.mime_type).strip().lower()
            if not updated_mime:
                raise ValueError("mime_type is required")
            conn.execute(
                """
                UPDATE dashboard_assets
                SET metadata_json = ?, mime_type = ?
                WHERE asset_id = ?
                """,
                (
                    json.dumps(current, separators=(",", ":"), sort_keys=True),
                    updated_mime,
                    asset_id,
                ),
            )
        updated = self.get(asset_id)
        assert updated is not None
        return updated

    # ------------------------------------------------------------------
    # Resolution and lifecycle
    # ------------------------------------------------------------------

    def resolve_asset(
        self,
        asset_id: str,
        *,
        expected_kind: DashboardAssetKind | str | None = None,
        owner_id: str | None = None,
        job_id: str | None = None,
        now: datetime | None = None,
    ) -> tuple[DashboardAssetRecord, Path]:
        """Authorize an asset and return its record plus safe server path.

        Staged assets require their owner. Claimed assets may be resolved by
        owner (for authenticated previews) or by their exact claimed job. A
        caller must provide at least one of those scopes.
        """

        if owner_id is None and job_id is None:
            raise DashboardAssetAccessError("owner_id or job_id is required to resolve an asset")
        record = self.get(asset_id)
        if record is None:
            raise DashboardAssetNotFoundError(f"dashboard asset not found: {asset_id}")
        if owner_id is not None:
            self._require_owner(record, owner_id)
        if record.status == DashboardAssetStatus.STAGED:
            if owner_id is None:
                raise DashboardAssetAccessError("staged assets can only be resolved by their owner")
            if self._utc(now or utc_now()) >= self._utc(record.expires_at):
                self._mark_expired(asset_id)
                raise DashboardAssetStateError("dashboard asset has expired")
        elif record.status == DashboardAssetStatus.CLAIMED:
            if job_id is not None and record.claimed_job_id != job_id:
                raise DashboardAssetAccessError("asset is claimed by another job")
        else:
            raise DashboardAssetStateError("dashboard asset has expired")
        if expected_kind is not None and record.kind != DashboardAssetKind(expected_kind):
            raise DashboardAssetKindError(
                f"asset {asset_id} is {record.kind.value}, expected {DashboardAssetKind(expected_kind).value}"
            )
        path = self._storage_path(Path(record.storage_path), must_exist=True)
        if not path.is_file():
            raise DashboardAssetPathError("asset file is missing")
        return record, path

    # Friendly shorter alias for API/worker callers.
    resolve = resolve_asset

    def validate_animate_assets(
        self,
        options: WanAnimateJobOptions,
        *,
        owner_id: str,
        job_id: str | None = None,
        now: datetime | None = None,
    ) -> list[DashboardAssetRecord]:
        """Validate ownership/kinds and metadata needed by Animate semantics."""

        requirements = collect_animate_asset_requirements(options)
        records: list[DashboardAssetRecord] = []
        for asset_id, kind in requirements.items():
            record, _ = self.resolve_asset(
                asset_id,
                expected_kind=kind,
                owner_id=owner_id,
                job_id=job_id,
                now=now,
            )
            records.append(record)
        driver = next(record for record in records if record.asset_id == options.driver.asset_id)
        if options.audio.mode in {"driver", "cast_voice"} and driver.metadata.get("has_audio") is not True:
            raise DashboardAssetMetadataError(
                f"{options.audio.mode.replace('_', ' ')} audio requires a probed driver audio stream"
            )
        duration = driver.metadata.get("duration_sec")
        if options.driver.timeline == "selected_range" and isinstance(duration, (int, float)):
            assert options.driver.end_sec is not None
            if options.driver.end_sec > float(duration):
                raise DashboardAssetMetadataError("selected driver range exceeds video duration")
        effective_duration = (
            float(options.driver.end_sec) - float(options.driver.start_sec)
            if options.driver.timeline == "selected_range"
            else float(duration or 0.0)
        )
        if effective_duration <= 0 or effective_duration > WAN_ANIMATE_MAX_DRIVER_RANGE_SEC:
            raise DashboardAssetMetadataError(
                "Animate driver range must be greater than 0 and at most "
                f"{WAN_ANIMATE_MAX_DRIVER_RANGE_SEC:.0f} seconds; select a shorter range"
            )
        return records

    def claim_assets(
        self,
        asset_ids: Iterable[str],
        *,
        owner_id: str,
        job_id: str,
        now: datetime | None = None,
    ) -> list[DashboardAssetRecord]:
        """Atomically claim all assets for one job or claim none of them."""

        ids = list(dict.fromkeys(asset_ids))
        if not ids:
            return []
        if not owner_id.strip() or not job_id.strip():
            raise ValueError("owner_id and job_id are required")
        for asset_id in ids:
            self._validate_asset_id(asset_id)
        now = self._utc(now or utc_now())
        placeholders = ",".join("?" for _ in ids)
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            rows = conn.execute(
                f"SELECT * FROM dashboard_assets WHERE asset_id IN ({placeholders})", ids
            ).fetchall()
            by_id = {row["asset_id"]: self._from_row(row) for row in rows}
            if missing := [asset_id for asset_id in ids if asset_id not in by_id]:
                raise DashboardAssetNotFoundError(f"dashboard asset not found: {missing[0]}")
            for asset_id in ids:
                record = by_id[asset_id]
                self._require_owner(record, owner_id)
                if record.status == DashboardAssetStatus.CLAIMED:
                    if record.claimed_job_id != job_id:
                        raise DashboardAssetStateError("asset is already claimed by another job")
                    continue  # idempotent retry for the same job
                if record.status == DashboardAssetStatus.EXPIRED or now >= self._utc(record.expires_at):
                    raise DashboardAssetStateError("dashboard asset has expired")
                if record.status != DashboardAssetStatus.STAGED:
                    raise DashboardAssetStateError("dashboard asset is not staged")
            conn.executemany(
                """
                UPDATE dashboard_assets
                SET status = ?, claimed_job_id = ?, claimed_at = ?
                WHERE asset_id = ? AND status = ?
                """,
                [
                    (
                        DashboardAssetStatus.CLAIMED.value,
                        job_id,
                        self._dt(now),
                        asset_id,
                        DashboardAssetStatus.STAGED.value,
                    )
                    for asset_id in ids
                    if by_id[asset_id].status == DashboardAssetStatus.STAGED
                ],
            )
        return [self._get_required(asset_id) for asset_id in ids]

    def release_claims(
        self,
        *,
        job_id: str,
        owner_id: str | None = None,
        asset_ids: Iterable[str] | None = None,
        extend_ttl: bool = True,
        now: datetime | None = None,
    ) -> list[DashboardAssetRecord]:
        """Roll back claims after queue creation fails.

        Only assets claimed by the exact job are touched. Supplying owner and
        IDs narrows this further, which is recommended for API rollback paths.
        """

        if not job_id.strip():
            raise ValueError("job_id is required")
        ids = list(dict.fromkeys(asset_ids)) if asset_ids is not None else None
        if ids == []:
            return []
        for asset_id in ids or []:
            self._validate_asset_id(asset_id)
        now = self._utc(now or utc_now())
        clauses = ["claimed_job_id = ?", "status = ?"]
        params: list[Any] = [job_id, DashboardAssetStatus.CLAIMED.value]
        if owner_id is not None:
            clauses.append("owner_id = ?")
            params.append(owner_id)
        if ids:
            clauses.append(f"asset_id IN ({','.join('?' for _ in ids)})")
            params.extend(ids)
        where = " AND ".join(clauses)
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            rows = conn.execute(f"SELECT * FROM dashboard_assets WHERE {where}", params).fetchall()
            matched_ids = [row["asset_id"] for row in rows]
            if matched_ids:
                expiry = self._dt(now + self.default_ttl) if extend_ttl else None
                if expiry is None:
                    conn.executemany(
                        """
                        UPDATE dashboard_assets
                        SET status = ?, claimed_job_id = NULL, claimed_at = NULL
                        WHERE asset_id = ?
                        """,
                        [(DashboardAssetStatus.STAGED.value, asset_id) for asset_id in matched_ids],
                    )
                else:
                    conn.executemany(
                        """
                        UPDATE dashboard_assets
                        SET status = ?, claimed_job_id = NULL, claimed_at = NULL, expires_at = ?
                        WHERE asset_id = ?
                        """,
                        [
                            (DashboardAssetStatus.STAGED.value, expiry, asset_id)
                            for asset_id in matched_ids
                        ],
                    )
        return [self._get_required(asset_id) for asset_id in matched_ids]

    # Rollback-oriented alias requested by API integration.
    unclaim_assets = release_claims

    def expire_staged(
        self,
        *,
        now: datetime | None = None,
        delete_files: bool = False,
    ) -> list[DashboardAssetRecord]:
        """Mark overdue staged assets expired and optionally remove their files."""

        now = self._utc(now or utc_now())
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            rows = conn.execute(
                """
                SELECT * FROM dashboard_assets
                WHERE status = ? AND expires_at <= ?
                """,
                (DashboardAssetStatus.STAGED.value, self._dt(now)),
            ).fetchall()
            records = [self._from_row(row) for row in rows]
            conn.execute(
                """
                UPDATE dashboard_assets SET status = ?
                WHERE status = ? AND expires_at <= ?
                """,
                (
                    DashboardAssetStatus.EXPIRED.value,
                    DashboardAssetStatus.STAGED.value,
                    self._dt(now),
                ),
            )
        expired = [self._get_required(record.asset_id) for record in records]
        if delete_files:
            for record in expired:
                try:
                    self._storage_path(Path(record.storage_path), must_exist=False).unlink(
                        missing_ok=True
                    )
                except OSError:
                    # A later cleanup run can retry; the asset is already
                    # inaccessible because its durable state is expired.
                    pass
        return expired

    def delete_staged(self, asset_id: str, *, owner_id: str) -> bool:
        """Delete an unclaimed staged/expired record and its managed file."""

        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT * FROM dashboard_assets WHERE asset_id = ?", (asset_id,)
            ).fetchone()
            if row is None:
                return False
            record = self._from_row(row)
            self._require_owner(record, owner_id)
            if record.status == DashboardAssetStatus.CLAIMED:
                raise DashboardAssetStateError("claimed assets cannot be deleted")
            conn.execute("DELETE FROM dashboard_assets WHERE asset_id = ?", (asset_id,))
        path = self._storage_path(Path(record.storage_path), must_exist=False)
        try:
            path.unlink(missing_ok=True)
        except OSError as exc:
            raise DashboardAssetPathError("asset record deleted but file cleanup failed") from exc
        return True

    def delete_claimed_for_job(self, job_id: str) -> list[DashboardAssetRecord]:
        """Delete immutable inputs after their terminal dashboard job is deleted."""

        if not job_id.strip():
            raise ValueError("job_id is required")
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            rows = conn.execute(
                """
                SELECT * FROM dashboard_assets
                WHERE status = ? AND claimed_job_id = ?
                """,
                (DashboardAssetStatus.CLAIMED.value, job_id),
            ).fetchall()
            records = [self._from_row(row) for row in rows]
            conn.execute(
                "DELETE FROM dashboard_assets WHERE status = ? AND claimed_job_id = ?",
                (DashboardAssetStatus.CLAIMED.value, job_id),
            )
        self._delete_record_files(records)
        return records

    def delete_orphaned_claims(self) -> list[DashboardAssetRecord]:
        """Repair a crash between asset claim and dashboard-job creation."""

        with self._connect() as conn:
            jobs_table = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'dashboard_jobs'"
            ).fetchone()
            if jobs_table is None:
                return []
            conn.execute("BEGIN IMMEDIATE")
            rows = conn.execute(
                """
                SELECT a.* FROM dashboard_assets AS a
                LEFT JOIN dashboard_jobs AS j ON j.job_id = a.claimed_job_id
                WHERE a.status = ? AND j.job_id IS NULL
                """,
                (DashboardAssetStatus.CLAIMED.value,),
            ).fetchall()
            records = [self._from_row(row) for row in rows]
            conn.executemany(
                "DELETE FROM dashboard_assets WHERE asset_id = ?",
                [(record.asset_id,) for record in records],
            )
        self._delete_record_files(records)
        return records

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_asset_id(asset_id: str) -> None:
        if _ASSET_ID_RE.fullmatch(asset_id) is None:
            raise DashboardAssetNotFoundError("invalid dashboard asset ID")

    @staticmethod
    def _safe_original_name(original_name: str) -> str:
        # Browsers may submit C:\\fakepath\\name.ext even on POSIX. Splitting
        # both separators ensures the value can never be treated as a path.
        normalized = original_name.replace("\\", "/").split("/")[-1]
        normalized = "".join(ch for ch in normalized if ch.isprintable()).strip()
        if not normalized or normalized in (".", ".."):
            return "upload"
        return normalized[:255]

    @staticmethod
    def _utc(value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("asset timestamps must be timezone-aware")
        return value.astimezone(timezone.utc)

    @staticmethod
    def _dt(value: datetime) -> str:
        return DashboardAssetStore._utc(value).isoformat()

    @staticmethod
    def _is_within(path: Path, root: Path) -> bool:
        try:
            path.relative_to(root)
            return True
        except ValueError:
            return False

    def _storage_path(self, path: Path, *, must_exist: bool) -> Path:
        try:
            resolved = Path(path).expanduser().resolve(strict=must_exist)
        except (FileNotFoundError, OSError) as exc:
            raise DashboardAssetPathError("asset storage path does not exist") from exc
        if not self._is_within(resolved, self.storage_root):
            raise DashboardAssetPathError("asset storage path escapes the managed root")
        return resolved

    @staticmethod
    def _require_owner(record: DashboardAssetRecord, owner_id: str) -> None:
        if not secrets.compare_digest(record.owner_id, owner_id):
            raise DashboardAssetAccessError("dashboard asset belongs to another owner")

    @staticmethod
    def _require_row(row: sqlite3.Row | None, asset_id: str) -> DashboardAssetRecord:
        if row is None:
            raise DashboardAssetNotFoundError(f"dashboard asset not found: {asset_id}")
        return DashboardAssetStore._from_row(row)

    def _get_required(self, asset_id: str) -> DashboardAssetRecord:
        record = self.get(asset_id)
        if record is None:
            raise DashboardAssetNotFoundError(f"dashboard asset not found: {asset_id}")
        return record

    def _mark_expired(self, asset_id: str) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE dashboard_assets SET status = ?
                WHERE asset_id = ? AND status = ?
                """,
                (
                    DashboardAssetStatus.EXPIRED.value,
                    asset_id,
                    DashboardAssetStatus.STAGED.value,
                ),
            )

    def _delete_record_files(self, records: Iterable[DashboardAssetRecord]) -> None:
        for record in records:
            try:
                self._storage_path(
                    Path(record.storage_path), must_exist=False
                ).unlink(missing_ok=True)
            except (OSError, DashboardAssetPathError):
                # The durable row is already gone; a later filesystem sweep can
                # remove a file held open by another local process.
                pass

    @staticmethod
    def _from_row(row: sqlite3.Row) -> DashboardAssetRecord:
        return DashboardAssetRecord(
            asset_id=row["asset_id"],
            owner_id=row["owner_id"],
            kind=DashboardAssetKind(row["kind"]),
            status=DashboardAssetStatus(row["status"]),
            original_name=row["original_name"],
            mime_type=row["mime_type"],
            sha256=row["sha256"],
            size_bytes=int(row["size_bytes"]),
            metadata=json.loads(row["metadata_json"] or "{}"),
            storage_path=row["storage_path"],
            created_at=datetime.fromisoformat(row["created_at"]),
            expires_at=datetime.fromisoformat(row["expires_at"]),
            claimed_job_id=row["claimed_job_id"],
            claimed_at=(datetime.fromisoformat(row["claimed_at"]) if row["claimed_at"] else None),
        )


__all__ = [
    "DashboardAssetAccessError",
    "DashboardAssetError",
    "DashboardAssetKindError",
    "DashboardAssetMetadataError",
    "DashboardAssetNotFoundError",
    "DashboardAssetPathError",
    "DashboardAssetQuotaError",
    "DashboardAssetStateError",
    "DashboardAssetStore",
    "collect_animate_asset_requirements",
    "make_dashboard_asset_id",
    "sha256_file",
]
