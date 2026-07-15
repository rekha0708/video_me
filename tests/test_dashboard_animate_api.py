"""Focused integration coverage for the dashboard's direct Animate API.

The media decoders are patched at their process boundary so these tests cover
HTTP ingestion, durable opaque assets, validation, and queue/claim ordering
without requiring ffprobe or Pillow fixtures.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest


try:
    import fastapi  # noqa: F401
    from fastapi.testclient import TestClient

    _HAS_FASTAPI = True
except ImportError:  # pragma: no cover - dashboard extras are optional
    _HAS_FASTAPI = False


pytestmark = pytest.mark.skipif(not _HAS_FASTAPI, reason="fastapi not installed")


@dataclass
class _Harness:
    client: Any
    repo: Any
    asset_store: Any
    config: Any
    normalized_inputs: list[tuple[Path, Path]]


def _make_harness(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    flux2_edit_enabled: bool = False,
    flux2_edit_max_references: int = 4,
    driver_has_audio: bool = True,
    flux_retarget_ready: bool = False,
    client_host: str = "testclient",
) -> _Harness:
    from core.config import AppConfig, Settings
    from core.models.profile import Cast, CastMember, ChannelProfile
    from core.wan_animate_readiness import WAN_ANIMATE_REQUIRED_MODEL_FILES
    from services import dashboard_api
    from services.dashboard_assets import DashboardAssetStore
    from services.dashboard_repository import DashboardRepository

    monkeypatch.delenv("VIDEO_ME_DASHBOARD_TOKEN", raising=False)
    monkeypatch.delenv("VIDEO_ME_DASHBOARD_ASSET_OWNER", raising=False)

    data_dir = tmp_path / "data"
    lora_dir = tmp_path / "loras"
    voice_dir = tmp_path / "voices"
    local_video_dir = tmp_path / "server-videos"
    wan_repo_dir = tmp_path / "Wan2.2"
    wan_model_dir = tmp_path / "Wan2.2-Animate-14B"
    wan_python = tmp_path / "venv" / "bin" / "python"
    for directory in (
        data_dir,
        lora_dir,
        voice_dir / "test_cast",
        local_video_dir,
        wan_repo_dir,
        wan_model_dir,
        wan_python.parent,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    wan_python.write_text("#!/bin/sh\n", encoding="utf-8")
    for relative in WAN_ANIMATE_REQUIRED_MODEL_FILES:
        checkpoint = wan_model_dir / relative
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(b"test-checkpoint")
    if flux_retarget_ready:
        flux_dir = wan_model_dir / "process_checkpoint/FLUX.1-Kontext-dev"
        for relative in (
            "model_index.json",
            "transformer/config.json",
            "text_encoder_2/config.json",
            "vae/config.json",
        ):
            path = flux_dir / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}", encoding="utf-8")
        for component in ("transformer", "text_encoder_2", "vae"):
            (flux_dir / component / "weights.safetensors").write_bytes(b"weights")
    (lora_dir / "test_cast_max.safetensors").write_bytes(b"test-lora")
    (voice_dir / "test_cast" / "max.wav").write_bytes(b"test-voice")

    settings = Settings(
        data_dir=data_dir,
        artifact_dir=tmp_path / "artifacts",
        sqlite_path=tmp_path / "dashboard.db",
        local_video_dir=local_video_dir,
        lora_dir=lora_dir,
        voice_dir=voice_dir,
        wan_animate_python=str(wan_python),
        wan_animate_repo_dir=wan_repo_dir,
        wan_animate_model_dir=wan_model_dir,
        wan_animate_data_root=data_dir,
        flux2_edit_enabled=flux2_edit_enabled,
        flux2_edit_max_references=flux2_edit_max_references,
    )
    config = AppConfig(
        settings=settings,
        channel_profile=ChannelProfile(
            id="test",
            name="Test",
            aspect_ratio="9:16",
            genre_content="education",
            tone="friendly",
            format="animated_character",
            made_for_kids=False,
        ),
        cast=Cast(
            id="test_cast",
            species="human",
            is_original_synthetic=True,
            members=[
                CastMember(
                    id="max",
                    name="Max",
                    visual_descriptor="adult test character",
                    lora_ref="loras/test_cast/max",
                    voice_profile_ref="voices/test_cast/max",
                    personality="friendly",
                )
            ],
        ),
    )

    async def fake_probe_video(path: Path, *, ffprobe_bin: str = "ffprobe") -> dict[str, Any]:
        assert path.is_file()
        assert ffprobe_bin == settings.ffprobe_bin
        return {
            "duration_sec": 6.25,
            "width": 1280,
            "height": 720,
            "fps": 30.0,
            "codec": "h264",
            "has_audio": driver_has_audio,
            "estimated_chunk_count": 1,
        }

    normalized_inputs: list[tuple[Path, Path]] = []

    def fake_normalize_image(source: Path, destination: Path) -> dict[str, Any]:
        assert source.is_file()
        normalized_inputs.append((source, destination))
        destination.write_bytes(b"normalized-png")
        return {"width": 768, "height": 1024, "format": "PNG"}

    monkeypatch.setattr(dashboard_api, "probe_video", fake_probe_video)
    monkeypatch.setattr(dashboard_api, "normalize_image", fake_normalize_image)

    repo = DashboardRepository(Path(settings.sqlite_path))
    app = dashboard_api.create_app(repository=repo, config_loader=lambda: config)
    client = TestClient(
        app,
        raise_server_exceptions=False,
        client=(client_host, 50000),
    )
    asset_store = DashboardAssetStore(
        repo.db_path,
        data_dir / "dashboard_assets",
        allowed_server_roots=(local_video_dir,),
    )
    return _Harness(client, repo, asset_store, config, normalized_inputs)


def _upload_video(harness: _Harness, *, name: str = "driver.mp4") -> dict[str, Any]:
    response = harness.client.post(
        "/api/assets/video/upload",
        files={"file": (name, b"fake-video-payload", "video/mp4")},
    )
    assert response.status_code == 200, response.text
    return response.json()["asset"]


def _upload_image(
    harness: _Harness,
    *,
    name: str = "reference.jpg",
    purpose: str = "garment",
) -> dict[str, Any]:
    response = harness.client.post(
        "/api/assets/image/upload",
        data={"purpose": purpose},
        files={"file": (name, b"fake-image-payload", "image/jpeg")},
    )
    assert response.status_code == 200, response.text
    return response.json()["asset"]


def _direct_job_payload(
    driver_asset_id: str,
    *,
    audio_mode: str = "driver",
    wardrobe_asset_ids: list[str] | None = None,
) -> dict[str, Any]:
    wardrobe_asset_ids = wardrobe_asset_ids or []
    character: dict[str, Any] = {
        "look_source": "auto_lora",
        "cast_ref": "test_cast",
        "member_id": "max",
    }
    if wardrobe_asset_ids:
        character = {
            "look_source": "styled_lora",
            "cast_ref": "test_cast",
            "member_id": "max",
            "wardrobe": {
                "clothing_type": "tailored jacket",
                "garment_asset_ids": wardrobe_asset_ids,
            },
        }
    audio: dict[str, Any] = {"mode": audio_mode}
    if audio_mode == "cast_voice":
        audio["voice_member_id"] = "max"
    return {
        "workflow_kind": "wan_animate_direct",
        "rights_cleared": True,
        "phase": "all",
        "animate": {
            "schema_version": 1,
            "mode": "animate",
            "driver": {"asset_id": driver_asset_id, "target_confirmed": True},
            "character": character,
            "audio": audio,
        },
    }


def _exact_image_job_payload(
    driver_asset_id: str,
    image_asset_id: str,
    *,
    audio_mode: str = "driver",
) -> dict[str, Any]:
    audio: dict[str, Any] = {"mode": audio_mode}
    if audio_mode == "cast_voice":
        # Structurally complete for AnimateAudioOptions, but invalid without a
        # selected cast/member that owns the requested voice.
        audio["voice_member_id"] = "max"
    return {
        "workflow_kind": "wan_animate_direct",
        "rights_cleared": True,
        "phase": "all",
        "animate": {
            "schema_version": 1,
            "mode": "animate",
            "driver": {"asset_id": driver_asset_id, "target_confirmed": True},
            "character": {
                "look_source": "exact_image",
                "exact_image_asset_id": image_asset_id,
            },
            "audio": audio,
        },
    }


def test_animate_options_exposes_readiness_cast_capabilities_and_feature_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _make_harness(
        tmp_path,
        monkeypatch,
        flux2_edit_enabled=True,
        flux2_edit_max_references=5,
        flux_retarget_ready=True,
    )

    response = harness.client.get("/api/animate/options")

    assert response.status_code == 200
    body = response.json()
    assert body["default"] == "test_cast"
    assert body["defaults"] == {"cast_ref": "test_cast"}
    assert body["readiness"]["ready"] is True
    assert body["readiness"]["wan_animate"]["ready"] is True
    assert body["features"]["flux2_edit_enabled"] is True
    assert body["features"]["flux2_edit_max_user_references"] == 4
    assert body["features"]["wan_flux_retarget_enabled"] is True
    test_cast = next(cast for cast in body["casts"] if cast["id"] == "test_cast")
    assert test_cast["members"] == [
        {
            "id": "max",
            "name": "Max",
            "visual_descriptor": "adult test character",
            "has_lora": True,
            "has_voice": True,
        }
    ]


def test_generated_look_rejects_renderer_that_cannot_apply_flux2_lora(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from core.models.dashboard import DashboardAssetStatus

    harness = _make_harness(tmp_path, monkeypatch)
    driver = _upload_video(harness)
    payload = _direct_job_payload(driver["asset_id"])
    payload["overrides"] = {"render_adapter": "comfyui_flux"}

    response = harness.client.post("/api/jobs", json=payload)

    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "FLUX2_RENDERER_REQUIRED"
    assert harness.asset_store.get(driver["asset_id"]).status == DashboardAssetStatus.STAGED
    assert harness.repo.list_jobs() == []


def test_flux_pose_retarget_rejects_job_until_optional_checkpoint_is_complete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _make_harness(tmp_path, monkeypatch, flux_retarget_ready=False)
    driver = _upload_video(harness)
    payload = _direct_job_payload(driver["asset_id"])
    payload["animate"]["advanced"] = {
        "retarget_pose": True,
        "use_flux_retarget": True,
    }

    response = harness.client.post("/api/jobs", json=payload)

    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "WAN_FLUX_RETARGET_NOT_READY"
    assert "--with-wan-animate-flux-retarget" in response.text
    assert harness.repo.list_jobs() == []


def test_video_upload_returns_opaque_asset_and_working_signed_preview(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _make_harness(tmp_path, monkeypatch)

    response = harness.client.post(
        "/api/assets/video/upload",
        files={"file": ("driver.mp4", b"fake-video-payload", "video/mp4")},
    )

    assert response.status_code == 200
    body = response.json()
    asset = body["asset"]
    assert asset["asset_id"].startswith("ast_")
    assert asset["original_name"] == "driver.mp4"
    assert asset["metadata"]["has_audio"] is True
    assert asset["media_url"].startswith(f"/api/assets/{asset['asset_id']}/media?token=")
    assert "storage_path" not in asset
    assert "owner_id" not in asset
    assert str(harness.config.settings.data_dir) not in response.text

    preview = harness.client.get(asset["media_url"])
    assert preview.status_code == 200
    assert preview.content == b"fake-video-payload"
    preview_path = f"/api/assets/{asset['asset_id']}/media"
    assert harness.client.get(preview_path).status_code == 403
    assert harness.client.get(
        preview_path,
        params={"token": "not-a-valid-token"},
    ).status_code == 403


def test_dashboard_write_routes_reject_non_loopback_clients_without_token(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _make_harness(
        tmp_path,
        monkeypatch,
        client_host="198.51.100.25",
    )

    response = harness.client.post(
        "/api/assets/video/upload",
        files={"file": ("driver.mp4", b"fake-video", "video/mp4")},
    )

    assert response.status_code == 403
    assert response.json()["detail"]["code"] == "LOCAL_DASHBOARD_ONLY"
    assert harness.client.get("/api/animate/options").status_code == 200


def test_image_upload_normalizes_before_registering_asset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _make_harness(tmp_path, monkeypatch)

    asset = _upload_image(harness, purpose="accessory")

    assert len(harness.normalized_inputs) == 1
    incoming, destination = harness.normalized_inputs[0]
    assert incoming.name.endswith(".jpg.incoming")
    assert not incoming.exists()
    assert destination.suffix == ".png"
    assert destination.read_bytes() == b"normalized-png"
    assert asset["mime_type"] == "image/png"
    assert asset["metadata"] == {
        "width": 768,
        "height": 1024,
        "format": "PNG",
        "purpose": "accessory",
    }
    assert "storage_path" not in asset


def test_direct_job_claims_assets_before_queue_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from core.models.dashboard import DashboardAssetStatus

    harness = _make_harness(tmp_path, monkeypatch)
    driver = _upload_video(harness)
    original_create = harness.repo.create_queued_job
    observed: dict[str, Any] = {}

    def create_after_claim(request: Any, *, priority: int = 100, job_id: str | None = None):
        assert job_id is not None
        record = harness.asset_store.get(driver["asset_id"])
        assert record is not None
        observed["status_at_queue"] = record.status
        observed["claimed_job_id_at_queue"] = record.claimed_job_id
        assert record.status == DashboardAssetStatus.CLAIMED
        assert record.claimed_job_id == job_id
        return original_create(request, priority=priority, job_id=job_id)

    monkeypatch.setattr(harness.repo, "create_queued_job", create_after_claim)

    response = harness.client.post("/api/jobs", json=_direct_job_payload(driver["asset_id"]))

    assert response.status_code == 200, response.text
    body = response.json()
    assert observed == {
        "status_at_queue": DashboardAssetStatus.CLAIMED,
        "claimed_job_id_at_queue": body["job_id"],
    }
    record = harness.asset_store.get(driver["asset_id"])
    assert record is not None
    assert record.status == DashboardAssetStatus.CLAIMED
    assert record.claimed_job_id == body["job_id"]
    job = harness.repo.get_job(body["job_id"])
    assert job is not None
    assert job.request["workflow_kind"] == "wan_animate_direct"
    assert job.request["animate"]["driver"]["asset_id"] == driver["asset_id"]
    assert len(harness.repo.list_queue(body["job_id"])) == 1


def test_text_directed_complete_look_queues_without_reference_edit_feature(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _make_harness(tmp_path, monkeypatch, flux2_edit_enabled=False)
    driver = _upload_video(harness)
    payload = _direct_job_payload(driver["asset_id"])
    payload["animate"]["character"] = {
        "look_source": "styled_lora",
        "cast_ref": "test_cast",
        "member_id": "max",
        "wardrobe": {
            "change_targets": ["jewelry", "bags", "footwear", "makeup"],
            "jewelry": ["gold earrings"],
            "bags": ["black clutch"],
            "footwear": "strappy sandals",
            "makeup": "berry lipstick",
            "details": "Preserve hair and facial identity",
        },
    }

    response = harness.client.post("/api/jobs", json=payload)

    assert response.status_code == 200, response.text
    job = harness.repo.get_job(response.json()["job_id"])
    assert job is not None
    wardrobe = job.request["animate"]["character"]["wardrobe"]
    assert wardrobe["change_targets"] == ["jewelry", "bags", "footwear", "makeup"]
    assert wardrobe["jewelry"] == ["gold earrings"]
    assert wardrobe["bags"] == ["black clutch"]
    assert wardrobe["footwear"] == "strappy sandals"
    assert wardrobe["makeup"] == "berry lipstick"


def test_exact_image_job_without_cast_or_member_queues_and_claims_both_assets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from core.models.dashboard import DashboardAssetStatus

    harness = _make_harness(tmp_path, monkeypatch)
    driver = _upload_video(harness)
    character = _upload_image(harness, name="character.png", purpose="character")

    response = harness.client.post(
        "/api/jobs",
        json=_exact_image_job_payload(driver["asset_id"], character["asset_id"]),
    )

    assert response.status_code == 200, response.text
    job_id = response.json()["job_id"]
    job = harness.repo.get_job(job_id)
    assert job is not None
    stored_character = job.request["animate"]["character"]
    assert stored_character["look_source"] == "exact_image"
    assert stored_character["cast_ref"] is None
    assert stored_character["member_id"] is None
    for asset_id in (driver["asset_id"], character["asset_id"]):
        record = harness.asset_store.get(asset_id)
        assert record is not None
        assert record.status == DashboardAssetStatus.CLAIMED
        assert record.claimed_job_id == job_id


def test_exact_image_job_without_cast_rejects_cast_voice_before_claiming_assets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from core.models.dashboard import DashboardAssetStatus

    harness = _make_harness(tmp_path, monkeypatch)
    driver = _upload_video(harness)
    character = _upload_image(harness, name="character.png", purpose="character")

    response = harness.client.post(
        "/api/jobs",
        json=_exact_image_job_payload(
            driver["asset_id"],
            character["asset_id"],
            audio_mode="cast_voice",
        ),
    )

    assert response.status_code == 422
    assert "cast_voice requires a selected cast and member" in response.text
    assert harness.asset_store.get(driver["asset_id"]).status == DashboardAssetStatus.STAGED
    assert harness.asset_store.get(character["asset_id"]).status == DashboardAssetStatus.STAGED
    assert harness.repo.list_jobs() == []


def test_direct_job_rejects_reference_images_while_flux_edit_is_disabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from core.models.dashboard import DashboardAssetStatus

    harness = _make_harness(tmp_path, monkeypatch, flux2_edit_enabled=False)
    driver = _upload_video(harness)
    garment = _upload_image(harness)

    response = harness.client.post(
        "/api/jobs",
        json=_direct_job_payload(
            driver["asset_id"], wardrobe_asset_ids=[garment["asset_id"]]
        ),
    )

    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "FLUX2_EDIT_NOT_READY"
    assert harness.asset_store.get(driver["asset_id"]).status == DashboardAssetStatus.STAGED
    assert harness.asset_store.get(garment["asset_id"]).status == DashboardAssetStatus.STAGED
    assert harness.repo.list_jobs() == []


def test_direct_job_rejects_more_than_configured_flux_reference_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _make_harness(
        tmp_path,
        monkeypatch,
        flux2_edit_enabled=True,
        # One slot is reserved for the canonical cast identity image.
        flux2_edit_max_references=2,
    )
    driver = _upload_video(harness)
    garments = [_upload_image(harness, name=f"look-{index}.jpg") for index in range(2)]

    response = harness.client.post(
        "/api/jobs",
        json=_direct_job_payload(
            driver["asset_id"],
            wardrobe_asset_ids=[asset["asset_id"] for asset in garments],
        ),
    )

    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "TOO_MANY_FLUX2_REFERENCES"
    assert harness.repo.list_jobs() == []


def test_direct_job_rejects_driver_audio_mode_when_probe_found_no_audio(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from core.models.dashboard import DashboardAssetStatus

    harness = _make_harness(tmp_path, monkeypatch, driver_has_audio=False)
    driver = _upload_video(harness)

    response = harness.client.post("/api/jobs", json=_direct_job_payload(driver["asset_id"]))

    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "INVALID_ASSET_METADATA"
    assert "audio requires a probed driver audio stream" in response.text
    assert harness.asset_store.get(driver["asset_id"]).status == DashboardAssetStatus.STAGED
    assert harness.repo.list_jobs() == []


def test_queue_failure_releases_claim_and_leaves_no_job(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from core.models.dashboard import DashboardAssetStatus

    harness = _make_harness(tmp_path, monkeypatch)
    driver = _upload_video(harness)
    observed_job_ids: list[str] = []

    def fail_queue_creation(request: Any, *, priority: int = 100, job_id: str | None = None):
        assert job_id is not None
        record = harness.asset_store.get(driver["asset_id"])
        assert record is not None and record.status == DashboardAssetStatus.CLAIMED
        assert record.claimed_job_id == job_id
        observed_job_ids.append(job_id)
        raise RuntimeError("simulated repository failure")

    monkeypatch.setattr(harness.repo, "create_queued_job", fail_queue_creation)

    response = harness.client.post("/api/jobs", json=_direct_job_payload(driver["asset_id"]))

    assert response.status_code == 500
    assert len(observed_job_ids) == 1
    record = harness.asset_store.get(driver["asset_id"])
    assert record is not None
    assert record.status == DashboardAssetStatus.STAGED
    assert record.claimed_job_id is None
    assert record.claimed_at is None
    assert harness.repo.list_jobs() == []


def test_direct_retry_phase_all_validates_claimed_assets_and_queues_versioned_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from core.models.dashboard import (
        CreateDashboardJobRequest,
        DashboardAssetStatus,
        DashboardJobStatus,
        DashboardQueueAction,
    )

    harness = _make_harness(tmp_path, monkeypatch)
    driver = _upload_video(harness)
    created = harness.client.post(
        "/api/jobs",
        json=_direct_job_payload(driver["asset_id"]),
    )
    assert created.status_code == 200, created.text
    job_id = created.json()["job_id"]
    harness.repo.update_job_status(job_id, DashboardJobStatus.COMPLETED, completed=True)
    original_queue_ids = [item.queue_id for item in harness.repo.list_queue(job_id)]

    response = harness.client.post(
        f"/api/jobs/{job_id}/retry",
        json={"phase": "all"},
    )

    assert response.status_code == 200, response.text
    assert response.json()["phase"] == "all"
    assert response.json()["status"] == "queued"
    retried = harness.repo.get_job(job_id)
    assert retried is not None and retried.status == DashboardJobStatus.QUEUED
    queues = harness.repo.list_queue(job_id)
    assert [item.queue_id for item in queues[:-1]] == original_queue_ids
    retry_item = queues[-1]
    assert retry_item.queue_id == response.json()["queue_id"]
    assert retry_item.action == DashboardQueueAction.RESUME
    validated = CreateDashboardJobRequest.model_validate(retry_item.payload)
    assert validated.workflow_kind == "wan_animate_direct"
    assert validated.phase == "all"
    assert validated.animate is not None
    assert validated.animate.driver.asset_id == driver["asset_id"]
    record = harness.asset_store.get(driver["asset_id"])
    assert record is not None
    assert record.status == DashboardAssetStatus.CLAIMED
    assert record.claimed_job_id == job_id


@pytest.mark.parametrize(
    "retry_body",
    [
        {"phase": "render"},
        {"phase": "all", "video_adapter": "wan"},
        {"render_mode": "source_audio"},
    ],
)
def test_direct_retry_rejects_pipeline_phase_or_overrides_without_state_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    retry_body: dict[str, Any],
) -> None:
    from core.models.dashboard import DashboardJobStatus

    harness = _make_harness(tmp_path, monkeypatch)
    driver = _upload_video(harness)
    created = harness.client.post(
        "/api/jobs",
        json=_direct_job_payload(driver["asset_id"]),
    )
    assert created.status_code == 200, created.text
    job_id = created.json()["job_id"]
    harness.repo.update_job_status(job_id, DashboardJobStatus.COMPLETED, completed=True)
    queue_ids_before = [item.queue_id for item in harness.repo.list_queue(job_id)]

    response = harness.client.post(f"/api/jobs/{job_id}/retry", json=retry_body)

    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "INVALID_ANIMATE_RETRY"
    unchanged = harness.repo.get_job(job_id)
    assert unchanged is not None and unchanged.status == DashboardJobStatus.COMPLETED
    assert [item.queue_id for item in harness.repo.list_queue(job_id)] == queue_ids_before


def test_direct_retry_revalidates_claimed_asset_files_before_mutating_job(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from core.models.dashboard import DashboardJobStatus

    harness = _make_harness(tmp_path, monkeypatch)
    driver = _upload_video(harness)
    created = harness.client.post(
        "/api/jobs",
        json=_direct_job_payload(driver["asset_id"]),
    )
    assert created.status_code == 200, created.text
    job_id = created.json()["job_id"]
    harness.repo.update_job_status(job_id, DashboardJobStatus.FAILED)
    queue_ids_before = [item.queue_id for item in harness.repo.list_queue(job_id)]
    _, driver_path = harness.asset_store.resolve(
        driver["asset_id"],
        owner_id="dashboard-local",
        job_id=job_id,
    )
    driver_path.unlink()

    response = harness.client.post(
        f"/api/jobs/{job_id}/retry",
        json={"phase": "all"},
    )

    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "INVALID_ASSET_PATH"
    unchanged = harness.repo.get_job(job_id)
    assert unchanged is not None and unchanged.status == DashboardJobStatus.FAILED
    assert [item.queue_id for item in harness.repo.list_queue(job_id)] == queue_ids_before


def test_legacy_preview_route_cannot_bypass_signed_dashboard_asset_media(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _make_harness(tmp_path, monkeypatch)
    image = _upload_image(harness)
    _, asset_path = harness.asset_store.resolve(
        image["asset_id"], owner_id="dashboard-local"
    )
    encoded_asset_path = base64.urlsafe_b64encode(str(asset_path).encode()).decode()

    blocked = harness.client.get(f"/img/{encoded_asset_path}")

    assert blocked.status_code == 403

    generated = Path(harness.config.settings.data_dir) / "jobs" / "job-safe" / "look.png"
    generated.parent.mkdir(parents=True, exist_ok=True)
    generated.write_bytes(b"generated-preview")
    encoded_generated_path = base64.urlsafe_b64encode(str(generated).encode()).decode()

    allowed = harness.client.get(f"/img/{encoded_generated_path}")

    assert allowed.status_code == 200
    assert allowed.content == b"generated-preview"
