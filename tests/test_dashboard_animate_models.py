"""Versioning and cross-field validation for direct Wan Animate jobs."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from core.models.dashboard import CreateDashboardJobRequest
from services.dashboard_repository import DashboardRepository


DRIVER_ID = "ast_abcdefghijklmnopqrstuvwxyz123456"
IMAGE_ID = "ast_abcdefghijklmnopqrstuvwxyz654321"


def _request(**animate_overrides) -> CreateDashboardJobRequest:
    animate = {
        "driver": {"asset_id": DRIVER_ID, "target_confirmed": True},
        "character": {
            "look_source": "auto_lora",
            "cast_ref": "lady_model",
            "member_id": "meera",
        },
    }
    animate.update(animate_overrides)
    return CreateDashboardJobRequest(
        workflow_kind="wan_animate_direct",
        rights_cleared=True,
        animate=animate,
    )


def test_direct_animate_synthesizes_legacy_source_for_repository() -> None:
    request = _request()

    assert request.source is not None
    assert request.source.kind == "animate"
    assert request.source.url == "animate://direct-input"
    assert request.animate is not None
    assert request.animate.schema_version == 1
    assert request.animate.output.preserve_aspect is True


def test_pipeline_contract_remains_backward_compatible() -> None:
    request = CreateDashboardJobRequest(
        source={"kind": "url", "url": "https://example.com/video"},
        rights_cleared=True,
    )

    assert request.workflow_kind == "pipeline"
    assert request.animate is None
    assert request.source is not None
    assert request.source.kind == "url"


def test_pipeline_still_requires_source() -> None:
    with pytest.raises(ValidationError, match="pipeline jobs require source"):
        CreateDashboardJobRequest(rights_cleared=True)


def test_direct_workflow_requires_nested_options() -> None:
    with pytest.raises(ValidationError, match="requires animate options"):
        CreateDashboardJobRequest(workflow_kind="wan_animate_direct", rights_cleared=True)


def test_pipeline_rejects_direct_options() -> None:
    with pytest.raises(ValidationError, match="animate options require"):
        CreateDashboardJobRequest(
            source={"kind": "url", "url": "https://example.com/video"},
            animate={
                "driver": {"asset_id": DRIVER_ID, "target_confirmed": True},
                "character": {"cast_ref": "cast", "member_id": "member"},
            },
        )


def test_direct_workflow_is_versioned() -> None:
    with pytest.raises(ValidationError):
        _request(schema_version=2)


def test_selected_range_requires_ordered_bounds() -> None:
    with pytest.raises(ValidationError, match="both start_sec and end_sec"):
        _request(
            driver={
                "asset_id": DRIVER_ID,
                "target_confirmed": True,
                "timeline": "selected_range",
            }
        )

    with pytest.raises(ValidationError, match="greater than start_sec"):
        _request(
            driver={
                "asset_id": DRIVER_ID,
                "target_confirmed": True,
                "timeline": "selected_range",
                "start_sec": 3,
                "end_sec": 2,
            }
        )


def test_full_driver_rejects_stale_range_values() -> None:
    with pytest.raises(ValidationError, match="only valid for selected_range"):
        _request(
            driver={
                "asset_id": DRIVER_ID,
                "target_confirmed": True,
                "start_sec": 0,
                "end_sec": 3,
            }
        )


@pytest.mark.parametrize("target_confirmed", [None, False])
def test_driver_requires_explicit_target_confirmation(target_confirmed: bool | None) -> None:
    driver = {"asset_id": DRIVER_ID}
    if target_confirmed is not None:
        driver["target_confirmed"] = target_confirmed

    with pytest.raises(ValidationError):
        _request(driver=driver)


def test_unimplemented_confirmed_track_is_rejected_at_validation() -> None:
    with pytest.raises(ValidationError):
        _request(
            driver={
                "asset_id": DRIVER_ID,
                "target_confirmed": True,
                "subject_selection": "confirmed_track",
                "subject_track_id": "track-1",
            }
        )


def test_auto_lora_requires_cast_and_member() -> None:
    with pytest.raises(ValidationError, match="require cast_ref and member_id"):
        _request(character={"look_source": "auto_lora", "cast_ref": "cast"})


def test_styled_lora_requires_actual_wardrobe_direction() -> None:
    with pytest.raises(ValidationError, match="non-empty wardrobe"):
        _request(
            character={
                "look_source": "styled_lora",
                "cast_ref": "cast",
                "member_id": "member",
                "wardrobe": {},
            }
        )

    request = _request(
        character={
            "look_source": "styled_lora",
            "cast_ref": "cast",
            "member_id": "member",
            "wardrobe": {
                "clothing_type": "tailored suit",
                "primary_color": "navy",
                "garment_asset_ids": [IMAGE_ID],
            },
        }
    )
    assert request.animate is not None
    assert request.animate.character.wardrobe is not None
    assert request.animate.character.wardrobe.primary_color == "navy"

    with pytest.raises(ValidationError, match="non-empty wardrobe"):
        _request(
            character={
                "look_source": "styled_lora",
                "cast_ref": "cast",
                "member_id": "member",
                "wardrobe": {"negative_constraints": "no logos"},
            }
        )


def test_styled_lora_accepts_complete_look_scope_and_details() -> None:
    request = _request(
        character={
            "look_source": "styled_lora",
            "cast_ref": "cast",
            "member_id": "member",
            "wardrobe": {
                "change_targets": ["makeup", "jewelry", "bags", "footwear", "makeup"],
                "jewelry": [" gold hoop earrings ", "layered necklace"],
                "bags": ["black clutch"],
                "footwear": "strappy sandals",
                "makeup": "deep berry lipstick",
                "details": "Keep the watch and existing hairstyle unchanged",
            },
        }
    )

    assert request.animate is not None
    wardrobe = request.animate.character.wardrobe
    assert wardrobe is not None
    assert wardrobe.change_targets == ["makeup", "jewelry", "bags", "footwear"]
    assert wardrobe.jewelry == ["gold hoop earrings", "layered necklace"]
    assert wardrobe.bags == ["black clutch"]
    assert wardrobe.makeup == "deep berry lipstick"


def test_styled_lora_accepts_makeup_direction_without_clothing() -> None:
    request = _request(
        character={
            "look_source": "styled_lora",
            "cast_ref": "cast",
            "member_id": "member",
            "wardrobe": {"makeup": "matte red lipstick; preserve everything else"},
        }
    )

    assert request.animate is not None
    assert request.animate.character.wardrobe is not None
    assert request.animate.character.wardrobe.has_direction() is True


def test_complete_look_rejects_unknown_change_target() -> None:
    with pytest.raises(ValidationError):
        _request(
            character={
                "look_source": "styled_lora",
                "cast_ref": "cast",
                "member_id": "member",
                "wardrobe": {"change_targets": ["tattoo"]},
            }
        )


def test_complete_look_reference_images_require_consistent_scope() -> None:
    accessory_id = "ast_accessoryreferenceabcdefghijkl"
    with pytest.raises(ValidationError, match="styling-detail reference images require"):
        _request(
            character={
                "look_source": "styled_lora",
                "cast_ref": "cast",
                "member_id": "member",
                "wardrobe": {"accessory_asset_ids": [accessory_id]},
            }
        )

    with pytest.raises(ValidationError, match="require clothing in change_targets"):
        _request(
            character={
                "look_source": "styled_lora",
                "cast_ref": "cast",
                "member_id": "member",
                "wardrobe": {
                    "change_targets": ["makeup"],
                    "garment_asset_ids": [IMAGE_ID],
                },
            }
        )

    with pytest.raises(ValidationError, match="styling-detail reference images require"):
        _request(
            character={
                "look_source": "styled_lora",
                "cast_ref": "cast",
                "member_id": "member",
                "wardrobe": {
                    "change_targets": ["clothing"],
                    "details": "gold earrings from the reference",
                    "accessory_asset_ids": [accessory_id],
                },
            }
        )


def test_complete_look_rejects_reference_with_conflicting_roles() -> None:
    with pytest.raises(ValidationError, match="cannot be both clothing and styling detail"):
        _request(
            character={
                "look_source": "styled_lora",
                "cast_ref": "cast",
                "member_id": "member",
                "wardrobe": {
                    "change_targets": ["clothing", "jewelry"],
                    "garment_asset_ids": [IMAGE_ID],
                    "accessory_asset_ids": [IMAGE_ID],
                },
            }
        )


def test_complete_look_rejects_fields_outside_explicit_change_scope() -> None:
    with pytest.raises(ValidationError, match="directions outside change_targets: hair"):
        _request(
            character={
                "look_source": "styled_lora",
                "cast_ref": "cast",
                "member_id": "member",
                "wardrobe": {
                    "change_targets": ["makeup"],
                    "makeup": "berry lipstick",
                    "hair": "low bun",
                },
            }
        )


def test_exact_image_requires_an_image_asset() -> None:
    with pytest.raises(ValidationError, match="requires exact_image_asset_id"):
        _request(character={"look_source": "exact_image"})

    request = _request(
        character={"look_source": "exact_image", "exact_image_asset_id": IMAGE_ID},
        audio={"mode": "none"},
    )
    assert request.animate is not None
    assert request.animate.character.cast_ref is None


def test_cast_voice_requires_selected_matching_member() -> None:
    with pytest.raises(ValidationError, match="requires voice_member_id"):
        _request(audio={"mode": "cast_voice"})

    with pytest.raises(ValidationError, match="must match"):
        _request(audio={"mode": "cast_voice", "voice_member_id": "someone_else"})

    request = _request(audio={"mode": "cast_voice", "voice_member_id": "meera"})
    assert request.animate is not None
    assert request.animate.audio.timing == "match_driver"


def test_replace_rejects_pose_retargeting() -> None:
    with pytest.raises(ValidationError, match="does not support pose retargeting"):
        _request(mode="replace", advanced={"retarget_pose": True})


def test_motion_transfer_rejects_replacement_mask_controls() -> None:
    with pytest.raises(ValidationError, match="does not support replacement masks"):
        _request(advanced={"mask_iterations": 3, "mask_kernel": 7})


def test_replacement_accepts_bounded_odd_mask_controls() -> None:
    request = _request(
        mode="replace",
        advanced={
            "mask_iterations": 3,
            "mask_kernel": 7,
            "mask_w_len": 1,
            "mask_h_len": 1,
        },
    )

    assert request.animate is not None
    assert request.animate.advanced.mask_kernel == 7


def test_output_contract_preserves_aspect_and_has_factual_options() -> None:
    request = _request(
        output={
            "generation_area": "480p",
            "export": "scale_1080p",
            "preserve_aspect": True,
            "target_fps": 48,
        }
    )
    assert request.animate is not None
    assert request.animate.output.target_fps == 48

    with pytest.raises(ValidationError):
        _request(output={"preserve_aspect": False})


def test_lipsync_rejects_silent_export() -> None:
    with pytest.raises(ValidationError, match="lip-sync requires"):
        _request(audio={"mode": "none"}, lipsync={"enabled": True})


def test_direct_workflow_rejects_pipeline_phase() -> None:
    with pytest.raises(ValidationError, match="requires phase='all'"):
        CreateDashboardJobRequest(
            workflow_kind="wan_animate_direct",
            phase="render",
            animate={
                "driver": {"asset_id": DRIVER_ID, "target_confirmed": True},
                "character": {"cast_ref": "cast", "member_id": "member"},
            },
        )


def test_direct_request_survives_repository_queue_round_trip(tmp_path: Path) -> None:
    repository = DashboardRepository(tmp_path / "dashboard.sqlite3")
    request = _request(audio={"mode": "none"})

    job, queued = repository.create_queued_job(request)
    claimed = repository.claim_next_action("animate-worker")

    assert job.source_kind == "animate"
    assert queued.payload["workflow_kind"] == "wan_animate_direct"
    assert claimed is not None
    restored = CreateDashboardJobRequest(**claimed.payload)
    assert restored.animate is not None
    assert restored.animate.driver.asset_id == DRIVER_ID
