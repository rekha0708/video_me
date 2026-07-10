"""Unit tests for MusubiFluxAdapter param-driven behavior (no subprocess)."""
import asyncio
from pathlib import Path

import pytest

from adapters.render_character.musubi_flux_adapter import MusubiFluxAdapter
from core.models.capabilities import RenderCharacterRequest
from core.models.profile import CastMember


def _member() -> CastMember:
    return CastMember(
        id="max", name="Max", gender="boy",
        visual_descriptor="cartoon boy in striped shirt",
        lora_ref="loras/kids_duo/max", voice_profile_ref="voices/kids_duo/max",
        personality="eager", signature_expressions=["grin"],
    )


def _req(tmp_path: Path, **kwargs) -> RenderCharacterRequest:
    return RenderCharacterRequest(
        member=kwargs.get("member", _member()),
        setting=kwargs.get("setting", "cozy kitchen"),
        shot_id=kwargs.get("shot_id", "s01"),
        camera=kwargs.get("camera", "close-up"),
        lora_file=kwargs.get("lora_file", ""),
        trigger=kwargs.get("trigger", ""),
        style_suffix=kwargs.get("style_suffix", ""),
    )


def _adapter(tmp_path: Path) -> MusubiFluxAdapter:
    return MusubiFluxAdapter(work_dir=tmp_path / "renders", lora_dir=tmp_path / "loras")


def test_build_prompt_includes_camera_setting_trigger(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    prompt = adapter._build_prompt(_req(tmp_path, trigger="mxtok"), skip_lora=False)
    assert prompt.startswith("mxtok")               # trigger first
    assert "cozy kitchen" in prompt
    assert "close-up shot" in prompt


def test_build_prompt_without_trigger(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    prompt = adapter._build_prompt(_req(tmp_path), skip_lora=False)
    assert prompt.startswith("cartoon boy")


def test_build_prompt_defaults_to_cartoon_style_when_unset(tmp_path: Path) -> None:
    """No style_suffix on the request (e.g. cast has no params.py) → cartoon default."""
    adapter = _adapter(tmp_path)
    prompt = adapter._build_prompt(_req(tmp_path), skip_lora=False)
    assert "children's animation style, cartoon" in prompt


def test_build_prompt_uses_cast_style_suffix_when_set(tmp_path: Path) -> None:
    """A cast's params.py can override the style (e.g. photorealistic LoRAs)."""
    adapter = _adapter(tmp_path)
    prompt = adapter._build_prompt(
        _req(tmp_path, style_suffix="photorealistic, cinematic lighting"),
        skip_lora=False,
    )
    assert "photorealistic, cinematic lighting" in prompt
    assert "children's animation style" not in prompt


def test_check_lora_prefers_params_lora_file(tmp_path: Path) -> None:
    lora_dir = tmp_path / "loras"
    lora_dir.mkdir()
    (lora_dir / "kids_duo_max.safetensors").write_bytes(b"weights")
    adapter = _adapter(tmp_path)
    path = adapter._check_lora(_req(tmp_path, lora_file="kids_duo_max.safetensors"))
    assert path.name == "kids_duo_max.safetensors"


def test_check_lora_params_file_missing_raises(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    with pytest.raises(RuntimeError, match="params.py"):
        adapter._check_lora(_req(tmp_path, lora_file="ghost.safetensors"))


def test_check_lora_falls_back_to_lora_ref(tmp_path: Path) -> None:
    lora_dir = tmp_path / "loras"
    lora_dir.mkdir()
    (lora_dir / "kids_duo_max.safetensors").write_bytes(b"weights")
    adapter = _adapter(tmp_path)
    # No lora_file on the request → derive from member.lora_ref.
    path = adapter._check_lora(_req(tmp_path, lora_file=""))
    assert path.name == "kids_duo_max.safetensors"


# ── Multi-character prompt tests ─────────────────────────────────────────────


def _other_member() -> CastMember:
    return CastMember(
        id="zoe", name="Zoe", gender="girl",
        visual_descriptor="cartoon girl with pink bow and yellow dress",
        lora_ref="loras/kids_duo/zoe", voice_profile_ref="voices/kids_duo/zoe",
        personality="playful", signature_expressions=["giggle"],
    )


def test_build_prompt_includes_other_member(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    req = RenderCharacterRequest(
        member=_member(),
        setting="cozy kitchen",
        shot_id="s01",
        camera="medium",
        other_members=[_other_member()],
    )
    prompt = adapter._build_prompt(req, skip_lora=False)
    assert "also present: cartoon girl with pink bow" in prompt
    assert "cartoon boy in striped shirt" in prompt


def test_build_prompt_no_other_members_unchanged(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    req_with = RenderCharacterRequest(
        member=_member(), setting="park", shot_id="s01", camera="wide",
        other_members=[],
    )
    req_without = RenderCharacterRequest(
        member=_member(), setting="park", shot_id="s01", camera="wide",
    )
    assert adapter._build_prompt(req_with, skip_lora=False) == \
           adapter._build_prompt(req_without, skip_lora=False)


# ── Batched subprocess tests (run / run_many via --from_file) ────────────────


class _FakeProc:
    def __init__(self, returncode: int = 0, stdout: bytes = b"") -> None:
        self.returncode = returncode
        self._stdout = stdout

    async def communicate(self):
        return self._stdout, None


def _install_fake_subprocess(monkeypatch, calls: list[list[str]], *, returncode: int = 0,
                             skip_seeds: set[int] | None = None) -> None:
    """Replace create_subprocess_exec with a fake that reads the prompts file
    and writes one musubi-named PNG per line into --save_path."""

    async def fake_exec(*cmd, **_kwargs):
        cmd = list(cmd)
        calls.append(cmd)
        if returncode == 0:
            save_path = Path(cmd[cmd.index("--save_path") + 1])
            prompts_file = Path(cmd[cmd.index("--from_file") + 1])
            for line in prompts_file.read_text().splitlines():
                if not line.strip():
                    continue
                seed = int(line.split(" --d ")[1].split()[0])
                if skip_seeds and seed in skip_seeds:
                    continue
                # musubi save_images_grid naming: {time_flag}_{seed}__{i:03d}.png
                (save_path / f"20990101-000000-000_{seed}__000.png").write_bytes(b"png")
        return _FakeProc(returncode=returncode, stdout=b"boom" if returncode else b"")

    import adapters.render_character.musubi_flux_adapter as mod
    monkeypatch.setattr(mod.asyncio, "create_subprocess_exec", fake_exec)


def _real_lora(tmp_path: Path, name: str = "kids_duo_max") -> None:
    lora_dir = tmp_path / "loras"
    lora_dir.mkdir(exist_ok=True)
    (lora_dir / f"{name}.safetensors").write_bytes(b"weights")


@pytest.mark.asyncio
async def test_run_batches_all_candidates_into_one_subprocess(
    tmp_path: Path, monkeypatch
) -> None:
    """3 candidates → ONE subprocess with a 3-line prompts file (one model load)."""
    _real_lora(tmp_path)
    adapter = MusubiFluxAdapter(
        work_dir=tmp_path / "renders", lora_dir=tmp_path / "loras", num_images=3
    )
    calls: list[list[str]] = []
    _install_fake_subprocess(monkeypatch, calls)

    result = await adapter.run(_req(tmp_path))

    assert len(calls) == 1
    cmd = calls[0]
    assert "--from_file" in cmd and "--prompt" not in cmd
    assert result.images == [
        str(tmp_path / "renders" / "s01" / "max" / f"render_{i:02d}.png") for i in range(3)
    ]
    for uri in result.images:
        assert Path(uri).exists()


@pytest.mark.asyncio
async def test_run_many_prompt_lines_have_unique_seeds_and_params(
    tmp_path: Path, monkeypatch
) -> None:
    _real_lora(tmp_path)
    adapter = MusubiFluxAdapter(
        work_dir=tmp_path / "renders", lora_dir=tmp_path / "loras", num_images=2
    )
    calls: list[list[str]] = []
    captured: dict[str, str] = {}

    async def fake_exec(*cmd, **_kwargs):
        cmd = list(cmd)
        calls.append(cmd)
        prompts_file = Path(cmd[cmd.index("--from_file") + 1])
        captured["prompts"] = prompts_file.read_text()
        save_path = Path(cmd[cmd.index("--save_path") + 1])
        for line in captured["prompts"].splitlines():
            seed = int(line.split(" --d ")[1].split()[0])
            (save_path / f"20990101-000000-000_{seed}__000.png").write_bytes(b"png")
        return _FakeProc()

    import adapters.render_character.musubi_flux_adapter as mod
    monkeypatch.setattr(mod.asyncio, "create_subprocess_exec", fake_exec)

    await adapter.run_many(
        [_req(tmp_path, shot_id="s01"), _req(tmp_path, shot_id="s02")]
    )

    lines = captured["prompts"].strip().splitlines()
    assert len(lines) == 4  # 2 shots × 2 candidates, one subprocess
    assert len(calls) == 1
    seeds = [int(l.split(" --d ")[1].split()[0]) for l in lines]
    assert seeds == [0, 1, 2, 3]  # globally unique per line → unambiguous file mapping
    for line in lines:
        assert " --s 20" in line and " --g 3.5" in line


@pytest.mark.asyncio
async def test_run_many_groups_by_lora(tmp_path: Path, monkeypatch) -> None:
    """Different members (different LoRAs) → one subprocess per LoRA group."""
    _real_lora(tmp_path, "kids_duo_max")
    _real_lora(tmp_path, "kids_duo_zoe")
    adapter = MusubiFluxAdapter(
        work_dir=tmp_path / "renders", lora_dir=tmp_path / "loras", num_images=1
    )
    calls: list[list[str]] = []
    _install_fake_subprocess(monkeypatch, calls)

    results = await adapter.run_many([
        _req(tmp_path, shot_id="s01"),
        RenderCharacterRequest(
            member=_other_member(), setting="park", shot_id="s02", camera="wide"
        ),
        _req(tmp_path, shot_id="s03"),
    ])

    assert len(calls) == 2  # max group (s01+s03) + zoe group (s02)
    loras = [cmd[cmd.index("--lora_weight") + 1] for cmd in calls]
    assert any("kids_duo_max" in l for l in loras)
    assert any("kids_duo_zoe" in l for l in loras)
    assert [r.member_id for r in results] == ["max", "zoe", "max"]
    for r in results:
        assert all(Path(u).exists() for u in r.images)


def test_sanitize_prompt_line_strips_newlines_and_dashes() -> None:
    out = MusubiFluxAdapter._sanitize_prompt_line("boy in shirt\n, in kitchen --extra")
    assert "\n" not in out
    assert "--" not in out
    assert "boy in shirt , in kitchen" in out


@pytest.mark.asyncio
async def test_run_many_missing_output_raises(tmp_path: Path, monkeypatch) -> None:
    _real_lora(tmp_path)
    adapter = MusubiFluxAdapter(
        work_dir=tmp_path / "renders", lora_dir=tmp_path / "loras", num_images=2
    )
    calls: list[list[str]] = []
    _install_fake_subprocess(monkeypatch, calls, skip_seeds={1})

    with pytest.raises(RuntimeError, match="no image for"):
        await adapter.run(_req(tmp_path))


@pytest.mark.asyncio
async def test_run_many_subprocess_failure_raises(tmp_path: Path, monkeypatch) -> None:
    _real_lora(tmp_path)
    adapter = _adapter(tmp_path)
    calls: list[list[str]] = []
    _install_fake_subprocess(monkeypatch, calls, returncode=1)

    with pytest.raises(RuntimeError, match="exit 1"):
        await adapter.run(_req(tmp_path))


class _FakeCancellableProc:
    """A subprocess whose communicate() hangs until cancelled, like a real
    musubi-tuner render being interrupted mid-model-load/inference."""

    def __init__(self) -> None:
        self.killed = False
        self.waited_after_kill = False

    async def communicate(self):
        try:
            await asyncio.Event().wait()  # never resolves on its own
        except asyncio.CancelledError:
            raise

    def kill(self) -> None:
        self.killed = True

    async def wait(self) -> None:
        self.waited_after_kill = self.killed


@pytest.mark.asyncio
async def test_run_kills_subprocess_on_cancellation(tmp_path: Path, monkeypatch) -> None:
    """A cancelled render (operator hits Cancel mid-render) must kill the child
    OS process, not just abandon the Python await — otherwise it keeps running
    and holding its GPU allocation forever, invisible to the dashboard, until
    it OOMs a later job's render (root cause of a real production incident)."""
    _real_lora(tmp_path)
    adapter = _adapter(tmp_path)
    fake_proc = _FakeCancellableProc()

    async def fake_exec(*cmd, **_kwargs):
        return fake_proc

    import adapters.render_character.musubi_flux_adapter as mod
    monkeypatch.setattr(mod.asyncio, "create_subprocess_exec", fake_exec)

    task = asyncio.ensure_future(adapter.run(_req(tmp_path)))
    await asyncio.sleep(0)  # let it reach the subprocess await point
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert fake_proc.killed is True
    assert fake_proc.waited_after_kill is True


def test_build_prompt_includes_shot_action(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    req = RenderCharacterRequest(
        member=_member(), setting="cozy kitchen", shot_id="s01",
        camera="close-up", action="stirs a bowl of batter with a big grin",
    )
    prompt = adapter._build_prompt(req, skip_lora=False)
    assert "stirs a bowl of batter" in prompt
    # action sits between setting and camera framing
    assert prompt.index("cozy kitchen") < prompt.index("stirs a bowl") < prompt.index("close-up shot")
