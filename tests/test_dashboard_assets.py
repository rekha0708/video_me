"""Security and lifecycle tests for the dashboard opaque asset store."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from core.models.dashboard import DashboardAssetKind, DashboardAssetStatus, WanAnimateJobOptions
from services.dashboard_assets import (
    DashboardAssetAccessError,
    DashboardAssetKindError,
    DashboardAssetMetadataError,
    DashboardAssetPathError,
    DashboardAssetQuotaError,
    DashboardAssetStateError,
    DashboardAssetStore,
    collect_animate_asset_requirements,
)


NOW = datetime(2026, 7, 14, 12, tzinfo=timezone.utc)


@pytest.fixture
def store(tmp_path: Path) -> DashboardAssetStore:
    server_root = tmp_path / "server-media"
    server_root.mkdir()
    return DashboardAssetStore(
        db_path=tmp_path / "dashboard.sqlite3",
        storage_root=tmp_path / "private-assets",
        allowed_server_roots=[server_root],
        default_ttl=timedelta(hours=1),
    )


def _stage(
    store: DashboardAssetStore,
    *,
    kind: DashboardAssetKind = DashboardAssetKind.VIDEO,
    owner_id: str = "session-a",
    metadata: dict | None = None,
    now: datetime = NOW,
):
    suffix = ".mp4" if kind == DashboardAssetKind.VIDEO else ".png"
    asset_id, path = store.allocate_path(kind, suffix=suffix)
    path.write_bytes(b"validated media bytes")
    return store.create_staged(
        owner_id=owner_id,
        kind=kind,
        original_name=f"../../user-file{suffix}",
        mime_type="video/mp4" if kind == DashboardAssetKind.VIDEO else "image/png",
        storage_path=path,
        asset_id=asset_id,
        metadata=metadata or {},
        now=now,
    )


def test_allocate_and_create_exposes_only_opaque_metadata(store: DashboardAssetStore) -> None:
    record = _stage(store)

    assert record.asset_id.startswith("ast_")
    assert record.original_name == "user-file.mp4"
    assert Path(record.storage_path).is_relative_to(store.storage_root)
    assert "storage_path" not in record.model_dump()
    assert record.sha256
    assert record.size_bytes == len(b"validated media bytes")


def test_create_rejects_file_outside_managed_root(store: DashboardAssetStore, tmp_path: Path) -> None:
    outside = tmp_path / "outside.mp4"
    outside.write_bytes(b"video")

    with pytest.raises(DashboardAssetPathError, match="escapes"):
        store.create_staged(
            owner_id="session-a",
            kind="video",
            original_name="video.mp4",
            mime_type="video/mp4",
            storage_path=outside,
            now=NOW,
        )


def test_total_asset_quota_is_enforced_transactionally(tmp_path: Path) -> None:
    limited = DashboardAssetStore(
        db_path=tmp_path / "dashboard.sqlite3",
        storage_root=tmp_path / "assets",
        max_total_bytes=30,
    )
    first = _stage(limited)

    with pytest.raises(DashboardAssetQuotaError, match="quota"):
        _stage(limited)

    assert limited.get(first.asset_id) is not None


def test_startup_cleanup_deletes_claims_without_a_dashboard_job(tmp_path: Path) -> None:
    from services.dashboard_repository import DashboardRepository

    db_path = tmp_path / "dashboard.sqlite3"
    DashboardRepository(db_path)  # creates the job table used by orphan repair
    orphan_store = DashboardAssetStore(db_path, tmp_path / "assets")
    record = _stage(orphan_store)
    path = Path(record.storage_path)
    orphan_store.claim_assets(
        [record.asset_id], owner_id="session-a", job_id="job-never-created", now=NOW
    )

    removed = orphan_store.delete_orphaned_claims()

    assert [item.asset_id for item in removed] == [record.asset_id]
    assert orphan_store.get(record.asset_id) is None
    assert not path.exists()


def test_resolution_enforces_owner_kind_and_claim_scope(store: DashboardAssetStore) -> None:
    record = _stage(store)

    with pytest.raises(DashboardAssetAccessError):
        store.resolve_asset(record.asset_id, owner_id="session-b", now=NOW)
    with pytest.raises(DashboardAssetKindError):
        store.resolve_asset(
            record.asset_id, owner_id="session-a", expected_kind="image", now=NOW
        )

    store.claim_assets([record.asset_id], owner_id="session-a", job_id="job-a", now=NOW)
    with pytest.raises(DashboardAssetAccessError, match="another job"):
        store.resolve_asset(record.asset_id, job_id="job-b")
    resolved, path = store.resolve_asset(record.asset_id, job_id="job-a")
    assert resolved.claimed_job_id == "job-a"
    assert path.exists()


def test_claim_is_atomic_when_one_asset_has_wrong_owner(store: DashboardAssetStore) -> None:
    first = _stage(store, owner_id="session-a")
    second = _stage(store, owner_id="session-b")

    with pytest.raises(DashboardAssetAccessError):
        store.claim_assets(
            [first.asset_id, second.asset_id],
            owner_id="session-a",
            job_id="job-a",
            now=NOW,
        )

    assert store.get(first.asset_id).status == DashboardAssetStatus.STAGED  # type: ignore[union-attr]
    assert store.get(second.asset_id).status == DashboardAssetStatus.STAGED  # type: ignore[union-attr]


def test_claim_is_idempotent_for_same_job_and_rejects_another(store: DashboardAssetStore) -> None:
    record = _stage(store)
    first = store.claim_assets(
        [record.asset_id], owner_id="session-a", job_id="job-a", now=NOW
    )
    second = store.claim_assets(
        [record.asset_id], owner_id="session-a", job_id="job-a", now=NOW
    )

    assert first[0].status == DashboardAssetStatus.CLAIMED
    assert second[0].claimed_job_id == "job-a"
    with pytest.raises(DashboardAssetStateError, match="another job"):
        store.claim_assets(
            [record.asset_id], owner_id="session-a", job_id="job-b", now=NOW
        )


def test_release_claims_supports_queue_rollback(store: DashboardAssetStore) -> None:
    record = _stage(store)
    store.claim_assets([record.asset_id], owner_id="session-a", job_id="job-a", now=NOW)

    released = store.release_claims(
        job_id="job-a",
        owner_id="session-a",
        asset_ids=[record.asset_id],
        now=NOW + timedelta(minutes=1),
    )

    assert released[0].status == DashboardAssetStatus.STAGED
    assert released[0].claimed_job_id is None
    assert released[0].expires_at == NOW + timedelta(hours=1, minutes=1)


def test_expire_staged_never_expires_claimed_assets(store: DashboardAssetStore) -> None:
    abandoned = _stage(store)
    claimed = _stage(store)
    store.claim_assets([claimed.asset_id], owner_id="session-a", job_id="job-a", now=NOW)

    expired = store.expire_staged(now=NOW + timedelta(hours=2))

    assert [item.asset_id for item in expired] == [abandoned.asset_id]
    assert store.get(abandoned.asset_id).status == DashboardAssetStatus.EXPIRED  # type: ignore[union-attr]
    assert store.get(claimed.asset_id).status == DashboardAssetStatus.CLAIMED  # type: ignore[union-attr]


def test_delete_staged_removes_file_but_refuses_claimed(store: DashboardAssetStore) -> None:
    staged = _stage(store)
    staged_path = Path(staged.storage_path)
    assert store.delete_staged(staged.asset_id, owner_id="session-a") is True
    assert not staged_path.exists()

    claimed = _stage(store)
    store.claim_assets([claimed.asset_id], owner_id="session-a", job_id="job-a", now=NOW)
    with pytest.raises(DashboardAssetStateError, match="cannot be deleted"):
        store.delete_staged(claimed.asset_id, owner_id="session-a")


def test_update_metadata_merges_probe_results(store: DashboardAssetStore) -> None:
    record = _stage(store, metadata={"codec": "h264"})

    updated = store.update_metadata(
        record.asset_id,
        owner_id="session-a",
        metadata={"duration_sec": 4.5, "has_audio": True},
        now=NOW,
    )

    assert updated.metadata == {
        "codec": "h264",
        "duration_sec": 4.5,
        "has_audio": True,
    }


def test_server_file_validation_blocks_traversal_and_symlink_escape(
    store: DashboardAssetStore, tmp_path: Path
) -> None:
    server_root = store.allowed_server_roots[0]
    allowed = server_root / "driver.mp4"
    allowed.write_bytes(b"video")
    assert store.validate_server_path("driver.mp4", expected_kind="video") == allowed

    outside = tmp_path / "secret.mp4"
    outside.write_bytes(b"secret")
    with pytest.raises(DashboardAssetPathError):
        store.validate_server_path(outside, expected_kind="video")

    link = server_root / "escape.mp4"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("symlinks are unavailable")
    with pytest.raises(DashboardAssetPathError):
        store.validate_server_path(link, expected_kind="video")


def test_collect_and_validate_animate_asset_requirements(store: DashboardAssetStore) -> None:
    driver = _stage(store, metadata={"duration_sec": 8.0, "has_audio": True})
    garment = _stage(store, kind=DashboardAssetKind.IMAGE)
    options = WanAnimateJobOptions(
        driver={
            "asset_id": driver.asset_id,
            "target_confirmed": True,
            "timeline": "selected_range",
            "start_sec": 1,
            "end_sec": 6,
        },
        character={
            "look_source": "styled_lora",
            "cast_ref": "cast",
            "member_id": "member",
            "wardrobe": {"garment_asset_ids": [garment.asset_id]},
        },
    )

    assert collect_animate_asset_requirements(options) == {
        driver.asset_id: DashboardAssetKind.VIDEO,
        garment.asset_id: DashboardAssetKind.IMAGE,
    }
    records = store.validate_animate_assets(options, owner_id="session-a", now=NOW)
    assert {record.asset_id for record in records} == {driver.asset_id, garment.asset_id}


def test_driver_audio_and_selected_range_use_probed_metadata(store: DashboardAssetStore) -> None:
    silent = _stage(store, metadata={"duration_sec": 4.0, "has_audio": False})
    driver_audio = WanAnimateJobOptions(
        driver={"asset_id": silent.asset_id, "target_confirmed": True},
        character={"cast_ref": "cast", "member_id": "member"},
    )
    with pytest.raises(DashboardAssetMetadataError, match="audio stream"):
        store.validate_animate_assets(driver_audio, owner_id="session-a", now=NOW)

    cast_voice = WanAnimateJobOptions(
        driver={"asset_id": silent.asset_id, "target_confirmed": True},
        character={"cast_ref": "cast", "member_id": "member"},
        audio={"mode": "cast_voice", "voice_member_id": "member"},
    )
    with pytest.raises(DashboardAssetMetadataError, match="driver audio stream"):
        store.validate_animate_assets(cast_voice, owner_id="session-a", now=NOW)

    no_audio = WanAnimateJobOptions(
        driver={
            "asset_id": silent.asset_id,
            "target_confirmed": True,
            "timeline": "selected_range",
            "start_sec": 1,
            "end_sec": 5,
        },
        character={"cast_ref": "cast", "member_id": "member"},
        audio={"mode": "none"},
    )
    with pytest.raises(DashboardAssetMetadataError, match="exceeds video duration"):
        store.validate_animate_assets(no_audio, owner_id="session-a", now=NOW)


def test_driver_range_is_hard_capped_at_thirty_seconds(store: DashboardAssetStore) -> None:
    driver = _stage(store, metadata={"duration_sec": 45.0, "has_audio": False})
    full = WanAnimateJobOptions(
        driver={"asset_id": driver.asset_id, "target_confirmed": True},
        character={"cast_ref": "cast", "member_id": "member"},
        audio={"mode": "none"},
    )
    with pytest.raises(DashboardAssetMetadataError, match="at most 30 seconds"):
        store.validate_animate_assets(full, owner_id="session-a", now=NOW)

    selected = full.model_copy(
        update={
            "driver": full.driver.model_copy(
                update={
                    "timeline": "selected_range",
                    "start_sec": 10.0,
                    "end_sec": 40.0,
                }
            )
        }
    )
    store.validate_animate_assets(selected, owner_id="session-a", now=NOW)
