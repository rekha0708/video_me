from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from core.animate_workflow import (
    AnimateWorkflowDependencies,
    ResolvedAnimateAsset,
    _assert_duration_close,
    _build_cast_voice,
    _character_provenance,
    _complete_look_change_targets,
    _directory_revision,
    _extract_audio,
    _render_canonical_look,
    _stable_wan_health,
    _validate_wan_health,
    _wardrobe_prompt,
    cleanup_wan_animate_processes,
    ensure_wan_animate_process_running,
    run_wan_animate_direct_job,
)
from core.models.capabilities import (
    AudioTrack,
    ImageApprovalResult,
    ImageSet,
    PreparedWanAnimateInput,
    TranscriptSegment,
    TranscribeResult,
    VideoClip,
)
from core.models.dashboard import CreateDashboardJobRequest, WanAnimateJobOptions, WardrobeSpec
from core.models.profile import CastMember


ASSET_ID = "ast_abcdefghijklmnopqrstuvwxyz123456"
IMAGE_ID = "ast_zyxwvutsrqponmlkjihgfedcba654321"


class FakeWanAnimate:
    version = "fake-wan-1"

    def __init__(self, root: Path) -> None:
        self.root = root
        self.prepare_calls = 0
        self.run_calls = 0

    async def prepare_inputs(self, requests):
        self.prepare_calls += 1
        prepared_dir = self.root / "prepared"
        prepared_dir.mkdir(parents=True, exist_ok=True)
        for name in ("src_ref.png", "src_pose.mp4", "src_face.mp4"):
            (prepared_dir / name).write_bytes(name.encode())
        return {
            requests[0].shot_id: PreparedWanAnimateInput(
                shot_id=requests[0].shot_id,
                prepared_dir=str(prepared_dir),
                driver_uri=requests[0].driver.uri,
                start_sec=requests[0].driver.start_sec,
                end_sec=requests[0].driver.end_sec,
                frame_count=120,
                fps=30,
                width=720,
                height=1280,
            )
        }

    async def run(self, request):
        self.run_calls += 1
        output = self.root / "raw.mp4"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"fake wan video")
        return VideoClip(uri=str(output), duration_sec=request.duration_sec, shot_id=request.shot_id)


class FakeRenderer:
    version = "fake-flux"

    def __init__(self, output: Path) -> None:
        self.output = output
        self.requests = []

    async def run(self, request):
        self.requests.append(request)
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.output.write_bytes(b"candidate")
        return ImageSet(images=[str(self.output)], member_id=request.member.id)


class FakeApproval:
    async def run(self, request):
        return ImageApprovalResult(
            approved_uris=[request.critique_results[0].candidate_uris[0]]
        )


class FakePassthroughMuseTalk:
    version = "fake-musetalk-1"

    def __init__(self) -> None:
        self.run_calls = 0

    @property
    def last_application_status(self) -> str:
        return "passthrough"

    async def health(self):
        return SimpleNamespace(status="ok", reason=None)

    async def run(self, request):
        self.run_calls += 1
        return VideoClip(uri=request.video_uri, duration_sec=4.0, shot_id=request.shot_id)


def test_complete_look_explicit_scope_is_authoritative() -> None:
    wardrobe = WardrobeSpec(
        change_targets=["makeup"],
        makeup="berry lipstick",
        details="Keep the hairstyle, watch, and bag unchanged",
    )

    assert _complete_look_change_targets(wardrobe) == ["makeup"]
    defensive_prompt = _wardrobe_prompt(
        SimpleNamespace(
            change_targets=["makeup"],
            makeup="berry lipstick",
            hair="low bun",
            details="",
        )
    )
    assert "makeup or lipstick: berry lipstick" in defensive_prompt
    assert "hair styling" not in defensive_prompt


def test_legacy_free_form_styling_does_not_get_a_restrictive_scope() -> None:
    wardrobe = WardrobeSpec(
        clothing_type="evening dress",
        accessories=["gold earrings", "clutch bag"],
        details="deep red lipstick",
    )

    assert _complete_look_change_targets(wardrobe) == []


def _direct_request(*, export: str = "generated") -> CreateDashboardJobRequest:
    return CreateDashboardJobRequest.model_validate(
        {
            "workflow_kind": "wan_animate_direct",
            "rights_cleared": True,
            "animate": {
                "driver": {"asset_id": ASSET_ID, "target_confirmed": True},
                "character": {
                    "look_source": "exact_image",
                    "exact_image_asset_id": IMAGE_ID,
                },
                "audio": {"mode": "none"},
                "output": {"export": export},
            },
        }
    )


@pytest.mark.asyncio
async def test_direct_exact_image_reuses_matching_preprocess_and_generation_manifests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    driver = tmp_path / "driver.mp4"
    reference = tmp_path / "reference.png"
    driver.write_bytes(b"driver")
    reference.write_bytes(b"reference")
    assets = {
        ASSET_ID: ResolvedAnimateAsset(ASSET_ID, "video", driver, "d" * 64),
        IMAGE_ID: ResolvedAnimateAsset(IMAGE_ID, "image", reference, "i" * 64),
    }

    def resolve(asset_id: str, expected_kind: str):
        assert assets[asset_id].kind == expected_kind
        return assets[asset_id]

    fake_video = FakeWanAnimate(tmp_path / "fake-video")
    deps = AnimateWorkflowDependencies(resolve_asset=resolve, video=fake_video)
    settings = SimpleNamespace(
        data_dir=tmp_path / "data",
        ffprobe_bin="ffprobe",
        ffmpeg_bin="ffmpeg",
        comfyui_base_url="http://localhost:8188",
        wan_animate_model_dir=tmp_path / "model",
        wan_animate_data_root=tmp_path / "data",
    )
    config = SimpleNamespace(settings=settings, cast=SimpleNamespace(id="unused", members=[]))

    monkeypatch.setattr(
        "core.animate_workflow.ensure_wan_animate_process_running",
        AsyncMock(return_value={"status": "ok", "flash_attn_3": True}),
    )
    monkeypatch.setattr(
        "core.animate_workflow._probe_media",
        AsyncMock(return_value={"duration_sec": 4.0, "has_audio": False, "video": {}, "audio": None}),
    )
    monkeypatch.setattr("core.animate_workflow._probe_duration", AsyncMock(return_value=4.0))
    monkeypatch.setattr("core.animate_workflow.free_comfyui", AsyncMock(return_value=True))
    monkeypatch.setattr("core.animate_workflow.prepare_video_model", AsyncMock())
    monkeypatch.setattr("core.animate_workflow.ensure_video_model_unloaded", AsyncMock())

    async def fake_export(source, output, *_):
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, output)
        return output

    export_mock = AsyncMock(side_effect=fake_export)
    monkeypatch.setattr("core.animate_workflow._export_video", export_mock)

    request = _direct_request()
    first = await run_wan_animate_direct_job(
        request, config, "job-direct", dependencies=deps
    )
    second = await run_wan_animate_direct_job(
        request, config, "job-direct", dependencies=deps
    )

    assert Path(first.final_video_uri).read_bytes() == b"fake wan video"
    assert second.final_video_uri == first.final_video_uri
    assert fake_video.prepare_calls == 1
    assert fake_video.run_calls == 1
    assert export_mock.await_count == 1
    manifests = Path(first.manifests_dir)
    assert (manifests / "canonical_look.json").is_file()
    assert (manifests / "animate_preprocess.json").is_file()
    assert (manifests / "animate_generate.json").is_file()
    assert (manifests / "animate_export.json").is_file()
    preprocess_inputs = json.loads(
        (manifests / "animate_preprocess.json").read_text(encoding="utf-8")
    )["inputs"]
    generation_inputs = json.loads(
        (manifests / "animate_generate.json").read_text(encoding="utf-8")
    )["inputs"]
    assert preprocess_inputs["preprocessor"]["version"] == "fake-wan-1"
    assert "model_revision" in preprocess_inputs
    assert generation_inputs["service"] == {
        "flash_attn_3": True,
        "status": "ok",
    }


@pytest.mark.asyncio
async def test_output_change_invalidates_export_but_not_wan_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    driver = tmp_path / "driver.mp4"
    reference = tmp_path / "reference.png"
    driver.write_bytes(b"driver")
    reference.write_bytes(b"reference")
    assets = {
        ASSET_ID: ResolvedAnimateAsset(ASSET_ID, "video", driver, "d" * 64),
        IMAGE_ID: ResolvedAnimateAsset(IMAGE_ID, "image", reference, "i" * 64),
    }
    fake_video = FakeWanAnimate(tmp_path / "fake-video")
    deps = AnimateWorkflowDependencies(
        resolve_asset=lambda asset_id, _kind: assets[asset_id], video=fake_video
    )
    settings = SimpleNamespace(
        data_dir=tmp_path / "data",
        ffprobe_bin="ffprobe",
        ffmpeg_bin="ffmpeg",
        comfyui_base_url="http://localhost:8188",
        wan_animate_model_dir=tmp_path / "model",
        wan_animate_data_root=tmp_path / "data",
    )
    config = SimpleNamespace(settings=settings, cast=SimpleNamespace(id="unused", members=[]))
    monkeypatch.setattr(
        "core.animate_workflow.ensure_wan_animate_process_running",
        AsyncMock(return_value={"status": "ok", "flash_attn_3": True}),
    )
    monkeypatch.setattr(
        "core.animate_workflow._probe_media",
        AsyncMock(return_value={"duration_sec": 4.0, "has_audio": False, "video": {}, "audio": None}),
    )
    monkeypatch.setattr("core.animate_workflow._probe_duration", AsyncMock(return_value=4.0))
    monkeypatch.setattr("core.animate_workflow.free_comfyui", AsyncMock(return_value=True))
    monkeypatch.setattr("core.animate_workflow.prepare_video_model", AsyncMock())
    monkeypatch.setattr("core.animate_workflow.ensure_video_model_unloaded", AsyncMock())

    async def fake_export(source, output, *_):
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(Path(source).read_bytes() + b" export")
        return output

    export_mock = AsyncMock(side_effect=fake_export)
    monkeypatch.setattr("core.animate_workflow._export_video", export_mock)

    await run_wan_animate_direct_job(
        _direct_request(export="generated"), config, "job-direct", dependencies=deps
    )
    await run_wan_animate_direct_job(
        _direct_request(export="scale_1080p"), config, "job-direct", dependencies=deps
    )

    assert fake_video.prepare_calls == 1
    assert fake_video.run_calls == 1
    assert export_mock.await_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("look_source", "wardrobe", "expected_fragments"),
    [
        ("auto_lora", None, ["entire body visible"]),
        (
            "styled_lora",
            {
                "change_targets": [
                    "clothing",
                    "jewelry",
                    "bags",
                    "footwear",
                    "makeup",
                    "hair",
                    "other",
                ],
                "clothing_type": "tailored suit",
                "primary_color": "emerald green",
                "jewelry": ["silver earrings"],
                "bags": ["structured handbag"],
                "footwear": "ankle boots",
                "makeup": "berry lipstick",
                "hair": "sleek low bun",
                "accessories": ["silver watch"],
                "details": "preserve the cast member's wedding ring",
                "negative_constraints": "no logos",
            },
            [
                "requested styling change scope",
                "tailored suit",
                "silver earrings",
                "structured handbag",
                "ankle boots",
                "berry lipstick",
                "sleek low bun",
                "silver watch",
                "preserve the cast member's wedding ring",
            ],
        ),
    ],
)
async def test_generated_look_modes_create_one_approved_canonical_reference(
    tmp_path: Path,
    look_source: str,
    wardrobe: dict | None,
    expected_fragments: list[str],
) -> None:
    from core.animate_workflow import _render_canonical_look

    member = CastMember(
        id="meera",
        name="Meera",
        visual_descriptor="adult Indian fashion model",
        lora_ref="loras/test/meera",
        voice_profile_ref="voices/test/meera",
        personality="confident",
    )
    character = {
        "look_source": look_source,
        "cast_ref": "test_cast",
        "member_id": member.id,
    }
    if wardrobe is not None:
        character["wardrobe"] = wardrobe
    options = WanAnimateJobOptions.model_validate(
        {
            "driver": {"asset_id": ASSET_ID, "target_confirmed": True},
            "character": character,
        }
    )
    renderer = FakeRenderer(tmp_path / "candidate.png")
    deps = AnimateWorkflowDependencies(
        resolve_asset=lambda *_: None,
        video=object(),
        render=renderer,
        image_approval=FakeApproval(),
    )
    config = SimpleNamespace(
        cast=SimpleNamespace(id="test_cast", members=[member]),
        settings=SimpleNamespace(flux2_edit_enabled=False),
    )

    result = await _render_canonical_look(options, config, deps, tmp_path / "job")

    assert result.read_bytes() == b"candidate"
    assert len(renderer.requests) == 1
    assert all(fragment in renderer.requests[0].action for fragment in expected_fragments)
    if wardrobe is not None:
        assert renderer.requests[0].negative_prompt == "no logos"


@pytest.mark.asyncio
async def test_complete_look_reference_edit_preserves_identity_and_labels_controls(
    tmp_path: Path,
) -> None:
    garment_id = "ast_garmentreferenceabcdefghijklmnop"
    accessory_id = "ast_accessoryreferenceabcdefghijkl"
    garment = tmp_path / "dress.png"
    accessory = tmp_path / "jewelry.png"
    garment.write_bytes(b"garment")
    accessory.write_bytes(b"accessory")
    member = CastMember(
        id="meera",
        name="Meera",
        visual_descriptor="adult Indian fashion model",
        lora_ref="loras/test/meera",
        voice_profile_ref="voices/test/meera",
        personality="confident",
    )
    options = WanAnimateJobOptions.model_validate(
        {
            "driver": {"asset_id": ASSET_ID, "target_confirmed": True},
            "character": {
                "look_source": "styled_lora",
                "cast_ref": "test_cast",
                "member_id": member.id,
                "wardrobe": {
                    "change_targets": ["clothing", "jewelry", "makeup"],
                    "clothing_type": "evening dress",
                    "jewelry": ["reference earrings"],
                    "makeup": "plum lipstick",
                    "details": "keep the hairstyle and bag unchanged",
                    "garment_asset_ids": [garment_id],
                    "accessory_asset_ids": [accessory_id],
                },
            },
        }
    )
    assets = {
        garment_id: ResolvedAnimateAsset(garment_id, "image", garment, "g" * 64),
        accessory_id: ResolvedAnimateAsset(
            accessory_id, "image", accessory, "a" * 64
        ),
    }
    renderer = FakeRenderer(tmp_path / "candidate.png")
    deps = AnimateWorkflowDependencies(
        resolve_asset=lambda asset_id, _: assets[asset_id],
        video=object(),
        render=renderer,
        image_approval=FakeApproval(),
    )
    config = SimpleNamespace(
        cast=SimpleNamespace(id="test_cast", members=[member]),
        settings=SimpleNamespace(
            flux2_edit_enabled=True,
            flux2_edit_max_references=4,
        ),
    )

    result = await _render_canonical_look(options, config, deps, tmp_path / "job")

    assert result.read_bytes() == b"candidate"
    assert len(renderer.requests) == 2
    edit = renderer.requests[1]
    assert edit.control_image_uris == [
        str(tmp_path / "candidate.png"),
        str(garment),
        str(accessory),
    ]
    assert "preserve the exact person's face" in edit.action
    assert "body proportions, hair" in edit.action
    assert "preserve every untargeted styling category" in edit.action
    assert "control image 1 is the cast identity" in edit.action
    assert "clothing or dress references" in edit.action
    assert "jewelry, bag, footwear, makeup" in edit.action
    assert "never copy another person's identity" in edit.action


@pytest.mark.asyncio
async def test_styled_lora_image_references_fail_closed_when_flux_edit_is_disabled(
    tmp_path: Path,
) -> None:
    from core.animate_workflow import _render_canonical_look

    garment_id = "ast_garmentreferenceabcdefghijklmnop"
    garment = tmp_path / "garment.png"
    garment.write_bytes(b"garment")
    member = CastMember(
        id="meera",
        name="Meera",
        visual_descriptor="adult Indian fashion model",
        lora_ref="loras/test/meera",
        voice_profile_ref="voices/test/meera",
        personality="confident",
    )
    options = WanAnimateJobOptions.model_validate(
        {
            "driver": {"asset_id": ASSET_ID, "target_confirmed": True},
            "character": {
                "look_source": "styled_lora",
                "cast_ref": "test_cast",
                "member_id": member.id,
                "wardrobe": {
                    "clothing_type": "evening dress",
                    "garment_asset_ids": [garment_id],
                },
            },
        }
    )
    renderer = FakeRenderer(tmp_path / "candidate.png")
    deps = AnimateWorkflowDependencies(
        resolve_asset=lambda *_: ResolvedAnimateAsset(
            garment_id, "image", garment, "g" * 64
        ),
        video=object(),
        render=renderer,
        image_approval=FakeApproval(),
    )
    config = SimpleNamespace(
        cast=SimpleNamespace(id="test_cast", members=[member]),
        settings=SimpleNamespace(flux2_edit_enabled=False),
    )

    with pytest.raises(RuntimeError, match="FLUX2_EDIT_ENABLED"):
        await _render_canonical_look(options, config, deps, tmp_path / "job")

    assert renderer.requests == []


@pytest.mark.asyncio
async def test_service_start_uses_configured_venv_model_and_data_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    python_bin = tmp_path / "venv" / "bin" / "python"
    python_bin.parent.mkdir(parents=True)
    python_bin.write_text("#!/bin/sh\n")
    wan_dir = tmp_path / "Wan2.2"
    model_dir = tmp_path / "Wan2.2-Animate-14B"
    data_root = tmp_path / "data"
    wan_dir.mkdir()
    model_dir.mkdir()
    data_root.mkdir()
    settings = SimpleNamespace(
        wan_animate_base_url="http://localhost:9133",
        wan_animate_python=str(python_bin),
        wan_animate_repo_dir=wan_dir,
        wan_animate_model_dir=model_dir,
        wan_animate_data_root=data_root,
        data_dir=data_root,
    )
    health = AsyncMock(
        side_effect=[
            None,
            {
                "status": "ok",
                "flash_attn_3": True,
                "require_flash_attn_3": True,
                "error": None,
                "model_dir": str(model_dir),
            },
        ]
    )
    spawn = AsyncMock(return_value=SimpleNamespace())
    monkeypatch.setattr("core.animate_workflow._wan_health_payload", health)
    monkeypatch.setattr("core.animate_workflow.asyncio.create_subprocess_exec", spawn)

    result = await ensure_wan_animate_process_running(
        settings, poll_sec=0, timeout_sec=1
    )

    assert result["status"] == "ok"
    args = spawn.await_args.args
    kwargs = spawn.await_args.kwargs
    assert args[:4] == (str(python_bin), "-m", "uvicorn", "services.wan_animate_server:app")
    assert args[-1] == "9133"
    assert kwargs["env"]["WAN_DIR"] == str(wan_dir)
    assert kwargs["env"]["WAN_ANIMATE_MODEL_DIR"] == str(model_dir)
    assert kwargs["env"]["WAN_ANIMATE_DATA_ROOT"] == str(data_root)
    assert kwargs["start_new_session"] is True


@pytest.mark.asyncio
async def test_unreachable_remote_service_is_not_replaced_by_local_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("core.animate_workflow._wan_health_payload", AsyncMock(return_value=None))
    spawn = AsyncMock()
    monkeypatch.setattr("core.animate_workflow.asyncio.create_subprocess_exec", spawn)
    settings = SimpleNamespace(wan_animate_base_url="https://gpu.example.test:8033")

    with pytest.raises(RuntimeError, match="Remote Wan Animate service is unreachable"):
        await ensure_wan_animate_process_running(settings, poll_sec=0, timeout_sec=0.01)

    spawn.assert_not_awaited()


@pytest.mark.asyncio
async def test_cancel_cleanup_terminates_preprocessor_and_inference_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    processes = [
        SimpleNamespace(wait=AsyncMock(return_value=0)),
        SimpleNamespace(wait=AsyncMock(return_value=0)),
    ]
    spawn = AsyncMock(side_effect=processes)
    monkeypatch.setattr("core.animate_workflow.asyncio.create_subprocess_exec", spawn)

    await cleanup_wan_animate_processes(kill_service=True)

    patterns = [call.args[2] for call in spawn.await_args_list]
    assert patterns == [
        "services.wan_animate_preprocess",
        "services.wan_animate_server:app",
    ]
    for process in processes:
        process.wait.assert_awaited_once()


def test_directory_revision_ignores_runtime_python_cache(tmp_path: Path) -> None:
    checkout = tmp_path / "wan"
    checkout.mkdir()
    (checkout / "config.json").write_text('{"revision": 1}', encoding="utf-8")
    before = _directory_revision(checkout)

    pycache = checkout / "pkg" / "__pycache__"
    pycache.mkdir(parents=True)
    (pycache / "module.cpython-313.pyc").write_bytes(b"runtime cache")

    assert _directory_revision(checkout) == before


def test_stable_wan_health_excludes_live_load_state() -> None:
    baseline = {
        "status": "ok",
        "flash_attn_3": True,
        "require_flash_attn_3": True,
        "model_dir": "/models/wan-animate",
        "model_loaded": False,
        "loading": False,
        "mode": None,
        "error": None,
    }
    loaded = {**baseline, "model_loaded": True, "mode": "animate"}

    assert _stable_wan_health(baseline) == _stable_wan_health(loaded)


def test_wan_health_rejects_wrong_checkpoint_identity(tmp_path: Path) -> None:
    expected = tmp_path / "expected-model"
    actual = tmp_path / "other-model"

    with pytest.raises(RuntimeError, match="checkpoint mismatch"):
        _validate_wan_health(
            {
                "status": "ok",
                "flash_attn_3": True,
                "require_flash_attn_3": True,
                "model_dir": str(actual),
            },
            expected_model_dir=expected,
        )


@pytest.mark.asyncio
async def test_styled_lora_reference_limit_fails_before_identity_render(
    tmp_path: Path,
) -> None:
    member = CastMember(
        id="meera",
        name="Meera",
        visual_descriptor="adult Indian fashion model",
        lora_ref="loras/test/meera",
        voice_profile_ref="voices/test/meera",
        personality="confident",
    )
    reference_ids = [
        "ast_referenceoneabcdefghijklmnop",
        "ast_referencetwoabcdefghijklmnop",
    ]
    options = WanAnimateJobOptions.model_validate(
        {
            "driver": {"asset_id": ASSET_ID, "target_confirmed": True},
            "character": {
                "look_source": "styled_lora",
                "cast_ref": "test_cast",
                "member_id": member.id,
                "wardrobe": {
                    "clothing_type": "evening dress",
                    "garment_asset_ids": reference_ids,
                },
            },
            "audio": {"mode": "none"},
        }
    )
    renderer = FakeRenderer(tmp_path / "candidate.png")
    deps = AnimateWorkflowDependencies(
        resolve_asset=lambda *_: pytest.fail("references must not resolve after precheck"),
        video=object(),
        render=renderer,
        image_approval=FakeApproval(),
    )
    config = SimpleNamespace(
        cast=SimpleNamespace(id="test_cast", members=[member]),
        settings=SimpleNamespace(
            flux2_edit_enabled=True,
            flux2_edit_max_references=2,
        ),
    )

    with pytest.raises(ValueError, match="upload at most 1"):
        await _render_canonical_look(options, config, deps, tmp_path / "job")

    assert renderer.requests == []


@pytest.mark.asyncio
async def test_character_provenance_hashes_member_fallback_lora(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    member = CastMember(
        id="meera",
        name="Meera",
        visual_descriptor="adult Indian fashion model",
        lora_ref="loras/test/meera",
        voice_profile_ref="voices/test/meera",
        personality="confident",
    )
    lora_dir = tmp_path / "loras"
    lora_dir.mkdir()
    lora = lora_dir / "test_meera.safetensors"
    lora.write_bytes(b"fallback lora")
    monkeypatch.setattr("core.animate_workflow.load_cast_params", lambda _cast_id: {})
    options = WanAnimateJobOptions.model_validate(
        {
            "driver": {"asset_id": ASSET_ID, "target_confirmed": True},
            "character": {
                "look_source": "auto_lora",
                "cast_ref": "test_cast",
                "member_id": member.id,
            },
            "audio": {"mode": "none"},
        }
    )
    config = SimpleNamespace(
        cast=SimpleNamespace(id="test_cast", members=[member]),
        settings=SimpleNamespace(lora_dir=lora_dir),
    )
    deps = AnimateWorkflowDependencies(resolve_asset=lambda *_: None, video=object())

    provenance = await _character_provenance(options, config, deps)

    assert provenance["lora"]["source"] == "member_fallback"
    assert provenance["lora"]["sha256"] == _sha256_file_for_test(lora)


def _sha256_file_for_test(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _cast_voice_options(member_id: str) -> WanAnimateJobOptions:
    return WanAnimateJobOptions.model_validate(
        {
            "driver": {"asset_id": ASSET_ID, "target_confirmed": True},
            "character": {
                "look_source": "auto_lora",
                "cast_ref": "test_cast",
                "member_id": member_id,
            },
            "audio": {"mode": "cast_voice", "voice_member_id": member_id},
        }
    )


def _cast_voice_config(tmp_path: Path, member: CastMember) -> SimpleNamespace:
    return SimpleNamespace(
        cast=SimpleNamespace(id="test_cast", members=[member]),
        settings=SimpleNamespace(ffmpeg_bin="ffmpeg", ffprobe_bin="ffprobe"),
    )


@pytest.mark.asyncio
async def test_cast_voice_unloads_whisper_when_transcription_fails(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.wav"
    source.write_bytes(b"audio")
    transcriber = SimpleNamespace(
        run=AsyncMock(side_effect=RuntimeError("transcription failed")),
        unload=AsyncMock(),
    )
    deps = AnimateWorkflowDependencies(
        resolve_asset=lambda *_: None,
        video=object(),
        transcriber=transcriber,
        voice=SimpleNamespace(),
    )

    with pytest.raises(RuntimeError, match="transcription failed"):
        await _build_cast_voice(
            source,
            3.0,
            SimpleNamespace(),
            SimpleNamespace(),
            deps,
            tmp_path / "job",
        )

    transcriber.unload.assert_awaited_once()


@pytest.mark.asyncio
async def test_cast_voice_rejects_overlapping_transcript_before_tts(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.wav"
    source.write_bytes(b"audio")
    transcript = TranscribeResult(
        segments=[
            TranscriptSegment(text="first", start=0.0, end=1.5),
            TranscriptSegment(text="second", start=1.0, end=2.0),
        ],
        language="en",
        full_text="first second",
    )
    transcriber = SimpleNamespace(
        run=AsyncMock(return_value=transcript), unload=AsyncMock()
    )
    voice = SimpleNamespace(run=AsyncMock())
    deps = AnimateWorkflowDependencies(
        resolve_asset=lambda *_: None,
        video=object(),
        transcriber=transcriber,
        voice=voice,
    )

    with pytest.raises(RuntimeError, match="overlapping speech segments"):
        await _build_cast_voice(
            source,
            3.0,
            SimpleNamespace(),
            SimpleNamespace(
                settings=SimpleNamespace(whisper_isolate_vocals=True)
            ),
            deps,
            tmp_path / "job",
        )

    transcriber.unload.assert_awaited_once()
    assert transcriber.run.await_args.args[0].isolate_vocals is True
    voice.run.assert_not_awaited()


@pytest.mark.asyncio
async def test_driver_audio_extraction_preserves_source_channels_and_pads_range(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "driver.mp4"
    source.write_bytes(b"video")
    output = tmp_path / "audio" / "driver.wav"
    captured: list[str] = []

    class Process:
        returncode = 0

        async def communicate(self):
            return b"", b""

    async def fake_exec(*command, **_kwargs):
        captured.extend(str(value) for value in command)
        return Process()

    monkeypatch.setattr(
        "core.animate_workflow.asyncio.create_subprocess_exec", fake_exec
    )

    await _extract_audio(source, output, 1.0, 4.25, "ffmpeg")

    assert "-ac" not in captured
    assert "-ar" not in captured
    audio_filter = captured[captured.index("-af") + 1]
    assert "apad" in audio_filter
    assert "atrim=0:3.250000" in audio_filter


def test_duration_guard_rejects_silent_truncation() -> None:
    _assert_duration_close("Wan Animate", 4.8, 5.0, 0.35)

    with pytest.raises(RuntimeError, match="silently truncated"):
        _assert_duration_close("Wan Animate", 3.9, 5.0, 0.35)


@pytest.mark.asyncio
async def test_generated_direct_workflow_rechecks_musubi_renderer_before_work(
    tmp_path: Path,
) -> None:
    request = CreateDashboardJobRequest.model_validate(
        {
            "workflow_kind": "wan_animate_direct",
            "rights_cleared": True,
            "animate": {
                "driver": {"asset_id": ASSET_ID, "target_confirmed": True},
                "character": {
                    "look_source": "auto_lora",
                    "cast_ref": "test_cast",
                    "member_id": "meera",
                },
                "audio": {"mode": "none"},
            },
        }
    )
    config = SimpleNamespace(
        settings=SimpleNamespace(
            render_adapter="comfyui_flux",
            data_dir=tmp_path,
        )
    )

    with pytest.raises(RuntimeError, match="render_adapter=musubi_flux"):
        await run_wan_animate_direct_job(
            request,
            config,
            "renderer-check",
            dependencies=AnimateWorkflowDependencies(
                resolve_asset=lambda *_: None,
                video=object(),
            ),
        )


@pytest.mark.asyncio
async def test_cast_voice_stops_fish_when_synthesis_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.wav"
    source.write_bytes(b"audio")
    member = CastMember(
        id="meera",
        name="Meera",
        visual_descriptor="adult Indian fashion model",
        lora_ref="loras/test/meera",
        voice_profile_ref="voices/test/meera",
        personality="confident",
    )
    transcript = TranscribeResult(
        segments=[TranscriptSegment(text="hello", start=0.0, end=1.0)],
        language="en",
        full_text="hello",
    )
    transcriber = SimpleNamespace(
        run=AsyncMock(return_value=transcript), unload=AsyncMock()
    )
    voice = SimpleNamespace(run=AsyncMock(side_effect=RuntimeError("tts failed")))
    deps = AnimateWorkflowDependencies(
        resolve_asset=lambda *_: None,
        video=object(),
        transcriber=transcriber,
        voice=voice,
    )
    prepare = AsyncMock()
    unload = AsyncMock()
    stop_fish = AsyncMock()
    monkeypatch.setattr("core.animate_workflow.load_cast_params", lambda _cast_id: {})
    monkeypatch.setattr("core.animate_workflow.prepare_voice_model", prepare)
    monkeypatch.setattr("core.animate_workflow.ensure_video_model_unloaded", unload)
    monkeypatch.setattr("core.animate_workflow.stop_fish_s2_process", stop_fish)

    with pytest.raises(RuntimeError, match="tts failed"):
        await _build_cast_voice(
            source,
            3.0,
            _cast_voice_options(member.id),
            _cast_voice_config(tmp_path, member),
            deps,
            tmp_path / "job",
        )

    prepare.assert_awaited_once()
    unload.assert_awaited_once_with(voice)
    stop_fish.assert_awaited_once()


@pytest.mark.asyncio
async def test_production_direct_job_clears_all_gpu_services_upfront(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    driver = tmp_path / "driver.mp4"
    reference = tmp_path / "reference.png"
    driver.write_bytes(b"driver")
    reference.write_bytes(b"reference")
    assets = {
        ASSET_ID: ResolvedAnimateAsset(ASSET_ID, "video", driver, "d" * 64),
        IMAGE_ID: ResolvedAnimateAsset(IMAGE_ID, "image", reference, "i" * 64),
    }
    fake_video = FakeWanAnimate(tmp_path / "fake-video")
    deps = AnimateWorkflowDependencies(
        resolve_asset=lambda asset_id, _kind: assets[asset_id], video=fake_video
    )
    settings = SimpleNamespace(
        data_dir=tmp_path / "data",
        ffprobe_bin="ffprobe",
        ffmpeg_bin="ffmpeg",
        comfyui_base_url="http://localhost:8188",
        wan_animate_model_dir=tmp_path / "model",
        wan_animate_data_root=tmp_path / "data",
    )
    config = SimpleNamespace(
        settings=settings, cast=SimpleNamespace(id="unused", members=[])
    )
    monkeypatch.setattr(
        "core.animate_workflow.build_default_dependencies", lambda **_kwargs: deps
    )
    monkeypatch.setattr(
        "core.animate_workflow.ensure_wan_animate_process_running",
        AsyncMock(return_value={"status": "ok", "flash_attn_3": True}),
    )
    monkeypatch.setattr(
        "core.animate_workflow._probe_media",
        AsyncMock(
            return_value={
                "duration_sec": 4.0,
                "has_audio": False,
                "video": {},
                "audio": None,
            }
        ),
    )
    monkeypatch.setattr("core.animate_workflow._probe_duration", AsyncMock(return_value=4.0))
    free = AsyncMock(return_value=True)
    unload = AsyncMock()
    stop_fish = AsyncMock()
    monkeypatch.setattr("core.animate_workflow.free_comfyui", free)
    monkeypatch.setattr("core.animate_workflow.ensure_video_model_unloaded", unload)
    monkeypatch.setattr("core.animate_workflow.stop_fish_s2_process", stop_fish)
    monkeypatch.setattr("core.animate_workflow.prepare_video_model", AsyncMock())

    async def fake_export(source_path, output, *_):
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_path, output)
        return output

    monkeypatch.setattr("core.animate_workflow._export_video", AsyncMock(side_effect=fake_export))

    await run_wan_animate_direct_job(
        _direct_request(), config, "production-job", asset_store=object()
    )

    # One reset before any pipeline work plus the normal Wan phase boundaries.
    assert unload.await_count == 2
    assert free.await_count == 2
    stop_fish.assert_awaited_once()


@pytest.mark.asyncio
async def test_requested_musetalk_passthrough_fails_direct_job_clearly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    driver = tmp_path / "driver.mp4"
    reference = tmp_path / "reference.png"
    driver.write_bytes(b"driver with audio")
    reference.write_bytes(b"reference")
    assets = {
        ASSET_ID: ResolvedAnimateAsset(ASSET_ID, "video", driver, "d" * 64),
        IMAGE_ID: ResolvedAnimateAsset(IMAGE_ID, "image", reference, "i" * 64),
    }
    fake_video = FakeWanAnimate(tmp_path / "fake-video")
    fake_lipsync = FakePassthroughMuseTalk()
    deps = AnimateWorkflowDependencies(
        resolve_asset=lambda asset_id, _kind: assets[asset_id],
        video=fake_video,
        lipsync=fake_lipsync,
    )
    settings = SimpleNamespace(
        data_dir=tmp_path / "data",
        ffprobe_bin="ffprobe",
        ffmpeg_bin="ffmpeg",
        comfyui_base_url="http://localhost:8188",
        wan_animate_model_dir=tmp_path / "model",
        wan_animate_data_root=tmp_path / "data",
    )
    config = SimpleNamespace(
        settings=settings, cast=SimpleNamespace(id="unused", members=[])
    )
    request = CreateDashboardJobRequest.model_validate(
        {
            "workflow_kind": "wan_animate_direct",
            "rights_cleared": True,
            "animate": {
                "driver": {"asset_id": ASSET_ID, "target_confirmed": True},
                "character": {
                    "look_source": "exact_image",
                    "exact_image_asset_id": IMAGE_ID,
                },
                "audio": {"mode": "driver"},
                "lipsync": {"enabled": True, "backend": "musetalk"},
            },
        }
    )
    monkeypatch.setattr(
        "core.animate_workflow.ensure_wan_animate_process_running",
        AsyncMock(return_value={"status": "ok", "flash_attn_3": True}),
    )
    monkeypatch.setattr(
        "core.animate_workflow._probe_media",
        AsyncMock(
            return_value={
                "duration_sec": 4.0,
                "has_audio": True,
                "video": {},
                "audio": {},
            }
        ),
    )
    monkeypatch.setattr("core.animate_workflow._probe_duration", AsyncMock(return_value=4.0))
    monkeypatch.setattr("core.animate_workflow.free_comfyui", AsyncMock(return_value=True))
    monkeypatch.setattr("core.animate_workflow.prepare_video_model", AsyncMock())
    monkeypatch.setattr("core.animate_workflow.ensure_video_model_unloaded", AsyncMock())

    async def fake_extract(_source, output, *_args):
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"driver audio")
        return output

    mux = AsyncMock()
    monkeypatch.setattr("core.animate_workflow._extract_audio", AsyncMock(side_effect=fake_extract))
    monkeypatch.setattr("core.animate_workflow._mux_audio", mux)

    with pytest.raises(
        RuntimeError,
        match=r"MuseTalk could not detect a usable face.*video unchanged",
    ):
        await run_wan_animate_direct_job(
            request, config, "musetalk-passthrough", dependencies=deps
        )

    assert fake_lipsync.run_calls == 1
    mux.assert_not_awaited()
    assert not (
        tmp_path
        / "data"
        / "jobs"
        / "musetalk-passthrough"
        / "animate_manifests"
        / "animate_lipsync.json"
    ).exists()
