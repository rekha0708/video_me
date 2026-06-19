import asyncio
import io
import sys
import wave
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from adapters.assemble_video.ffmpeg_adapter import (
    FfmpegAssembleAdapter,
    _DEFAULT_CAPTION_MARGIN,
    _DEFAULT_HEIGHT,
    _DEFAULT_WIDTH,
    _DEFAULT_WRAP_WIDTH,
)
from core.models.capabilities import (
    AssembleRequest,
    AudioTrack,
    FinalVideo,
    VideoClip,
)


# ------------------------------------------------------------------ helpers

def _make_wav_bytes(duration_sec: float = 3.0, sample_rate: int = 22050) -> bytes:
    num_frames = int(sample_rate * duration_sec)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * num_frames)
    return buf.getvalue()


def _write_clip(tmp_path: Path, name: str, duration: float = 2.5) -> VideoClip:
    path = tmp_path / name
    path.write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 40)
    return VideoClip(uri=str(path), duration_sec=duration, shot_id=name.rstrip(".mp4"))


def _write_audio(tmp_path: Path) -> AudioTrack:
    path = tmp_path / "dialogue.wav"
    path.write_bytes(_make_wav_bytes(5.0))
    return AudioTrack(uri=str(path), duration_sec=5.0, speaker_id=None)


def _request(tmp_path: Path, **kwargs) -> AssembleRequest:
    clips = kwargs.get("clips") or [
        _write_clip(tmp_path, "s01.mp4", 2.5),
        _write_clip(tmp_path, "s02.mp4", 3.0),
    ]
    audio = kwargs.get("audio") or _write_audio(tmp_path)
    return AssembleRequest(
        clips=clips,
        audio=audio,
        caption_text=kwargs.get(
            "caption_text", "How many apples? Let us count them! One two three four five!"
        ),
        aspect_ratio=kwargs.get("aspect_ratio", "9:16"),
        made_for_kids=kwargs.get("made_for_kids", True),
        disclosure_label_required=kwargs.get("disclosure_label_required", True),
    )


def _adapter(tmp_path: Path, **kwargs) -> FfmpegAssembleAdapter:
    return FfmpegAssembleAdapter(work_dir=tmp_path / "output", **kwargs)


async def _noop_ffmpeg(cmd: list[str]) -> None:
    """Fake _run_ffmpeg that writes the expected output file."""
    output = Path(cmd[-1])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(b"fake mp4")


# ------------------------------------------------------------------ _check_clips

def test_check_clips_passes_when_all_exist(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    clips = [_write_clip(tmp_path, "s01.mp4"), _write_clip(tmp_path, "s02.mp4")]
    adapter._check_clips(clips)  # no raise


def test_check_clips_raises_when_clip_missing(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    missing = VideoClip(uri=str(tmp_path / "missing.mp4"), duration_sec=2.0, shot_id="s99")
    with pytest.raises(FileNotFoundError) as exc_info:
        adapter._check_clips([missing])
    assert "lip_sync" in str(exc_info.value)
    assert "s99" in str(exc_info.value)


# ------------------------------------------------------------------ _write_concat_list

def test_write_concat_list_format(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    clips = [_write_clip(tmp_path, "s01.mp4"), _write_clip(tmp_path, "s02.mp4")]
    path = adapter._write_concat_list(clips, out_dir)
    lines = path.read_text().splitlines()
    assert len(lines) == 2
    for line in lines:
        assert line.startswith("file '")
        assert line.endswith("'")


def test_write_concat_list_uses_absolute_paths(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    clip = _write_clip(tmp_path, "s01.mp4")
    path = adapter._write_concat_list([clip], out_dir)
    content = path.read_text()
    assert content.startswith("file '/")  # absolute path


def test_write_concat_list_preserves_clip_order(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    c1 = _write_clip(tmp_path, "s01.mp4")
    c2 = _write_clip(tmp_path, "s02.mp4")
    c3 = _write_clip(tmp_path, "s03.mp4")
    path = adapter._write_concat_list([c1, c2, c3], out_dir)
    lines = path.read_text().splitlines()
    assert "s01" in lines[0]
    assert "s02" in lines[1]
    assert "s03" in lines[2]


# ------------------------------------------------------------------ _write_caption_file

def test_write_caption_file_creates_file(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    path = adapter._write_caption_file("Hello world", out_dir)
    assert path.exists()
    assert path.name == "caption.txt"


def test_write_caption_file_wraps_long_text(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    long_text = "word " * 50  # definitely longer than wrap width
    path = adapter._write_caption_file(long_text.strip(), out_dir)
    lines = path.read_text().splitlines()
    assert len(lines) > 1
    for line in lines:
        assert len(line) <= _DEFAULT_WRAP_WIDTH + 5  # small tolerance for long words


def test_write_caption_file_short_text_single_line(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    path = adapter._write_caption_file("Short text.", out_dir)
    assert path.read_text().strip() == "Short text."


# ------------------------------------------------------------------ _build_filter

def test_build_filter_contains_scale_pad(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    caption_file = tmp_path / "caption.txt"
    f = adapter._build_filter(caption_file, disclosure_required=False)
    assert f"scale={_DEFAULT_WIDTH}:{_DEFAULT_HEIGHT}" in f
    assert "pad=" in f
    assert "color=black" in f


def test_build_filter_contains_drawtext_for_caption(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    caption_file = tmp_path / "caption.txt"
    f = adapter._build_filter(caption_file, disclosure_required=False)
    assert "drawtext" in f
    assert "textfile=" in f
    assert str(caption_file) in f


def test_build_filter_includes_disclosure_when_required(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    caption_file = tmp_path / "caption.txt"
    f = adapter._build_filter(caption_file, disclosure_required=True)
    assert "AI" in f
    assert "Generated Content" in f


def test_build_filter_omits_disclosure_when_not_required(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    caption_file = tmp_path / "caption.txt"
    f = adapter._build_filter(caption_file, disclosure_required=False)
    assert "Generated Content" not in f


def test_build_filter_ends_with_output_label(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    caption_file = tmp_path / "caption.txt"
    for disclosure in (True, False):
        f = adapter._build_filter(caption_file, disclosure_required=disclosure)
        assert f.endswith("[v]")


# ------------------------------------------------------------------ _build_ffmpeg_args

def test_build_ffmpeg_args_includes_concat_input(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    req = _request(tmp_path)
    out = tmp_path / "output" / "final.mp4"
    concat = tmp_path / "concat.txt"
    audio = Path(req.audio.uri)
    caption = tmp_path / "caption.txt"
    args = adapter._build_ffmpeg_args(concat, audio, caption, out, req)
    assert "-f" in args and "concat" in args
    assert str(concat) in args


def test_build_ffmpeg_args_includes_audio_input(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    req = _request(tmp_path)
    out = tmp_path / "final.mp4"
    audio = Path(req.audio.uri)
    args = adapter._build_ffmpeg_args(
        tmp_path / "concat.txt", audio, tmp_path / "cap.txt", out, req
    )
    assert str(audio) in args


def test_build_ffmpeg_args_maps_processed_video_and_audio(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    req = _request(tmp_path)
    out = tmp_path / "final.mp4"
    args = adapter._build_ffmpeg_args(
        tmp_path / "concat.txt", Path(req.audio.uri),
        tmp_path / "cap.txt", out, req,
    )
    assert "[v]" in args
    assert "1:a" in args


def test_build_ffmpeg_args_includes_shortest_flag(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    req = _request(tmp_path)
    out = tmp_path / "final.mp4"
    args = adapter._build_ffmpeg_args(
        tmp_path / "concat.txt", Path(req.audio.uri),
        tmp_path / "cap.txt", out, req,
    )
    assert "-shortest" in args


def test_build_ffmpeg_args_output_is_final_mp4(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    req = _request(tmp_path)
    out = tmp_path / "output" / "final.mp4"
    args = adapter._build_ffmpeg_args(
        tmp_path / "concat.txt", Path(req.audio.uri),
        tmp_path / "cap.txt", out, req,
    )
    assert args[-1] == str(out)


def test_build_ffmpeg_args_uses_configured_crf(tmp_path: Path) -> None:
    adapter = FfmpegAssembleAdapter(work_dir=tmp_path / "o", crf=18)
    req = _request(tmp_path)
    out = tmp_path / "final.mp4"
    args = adapter._build_ffmpeg_args(
        tmp_path / "c.txt", Path(req.audio.uri),
        tmp_path / "cap.txt", out, req,
    )
    assert "-crf" in args
    assert "18" in args


# ------------------------------------------------------------------ _run_ffmpeg

async def test_run_ffmpeg_succeeds_on_zero_exit(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)

    mock_proc = MagicMock()
    mock_proc.returncode = 0
    mock_proc.communicate = AsyncMock(return_value=(b"", b""))

    with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=mock_proc)):
        await adapter._run_ffmpeg(["ffmpeg", "-version"])  # no raise


async def test_run_ffmpeg_raises_on_nonzero_exit(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)

    mock_proc = MagicMock()
    mock_proc.returncode = 1
    mock_proc.communicate = AsyncMock(return_value=(b"", b"error detail"))

    with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=mock_proc)):
        with pytest.raises(RuntimeError, match="ffmpeg exited 1"):
            await adapter._run_ffmpeg(["ffmpeg", "bad"])


async def test_run_ffmpeg_includes_stderr_in_error(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)

    mock_proc = MagicMock()
    mock_proc.returncode = 2
    mock_proc.communicate = AsyncMock(return_value=(b"", b"No such encoder: libx265"))

    with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=mock_proc)):
        with pytest.raises(RuntimeError, match="libx265"):
            await adapter._run_ffmpeg(["ffmpeg", "cmd"])


# ------------------------------------------------------------------ health

async def test_health_ok_when_ffmpeg_available(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)

    mock_proc = MagicMock()
    mock_proc.returncode = 0
    mock_proc.communicate = AsyncMock(return_value=(b"ffmpeg version ...", b""))

    with patch("shutil.which", return_value="/usr/bin/ffmpeg"), \
         patch("asyncio.create_subprocess_exec", AsyncMock(return_value=mock_proc)):
        health = await adapter.health()

    assert health.status == "ok"


async def test_health_down_when_ffmpeg_missing(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    with patch("shutil.which", return_value=None):
        health = await adapter.health()
    assert health.status == "down"
    assert "ffmpeg" in (health.reason or "").lower()


async def test_health_down_when_ffmpeg_fails(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)

    mock_proc = MagicMock()
    mock_proc.returncode = 1
    mock_proc.communicate = AsyncMock(return_value=(b"", b""))

    with patch("shutil.which", return_value="/usr/bin/ffmpeg"), \
         patch("asyncio.create_subprocess_exec", AsyncMock(return_value=mock_proc)):
        health = await adapter.health()

    assert health.status == "down"


# ------------------------------------------------------------------ run

async def test_run_returns_final_video(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    req = _request(tmp_path)

    with patch.object(adapter, "_run_ffmpeg", side_effect=_noop_ffmpeg):
        result = await adapter.run(req)

    assert isinstance(result, FinalVideo)
    assert result.uri.endswith("final.mp4")
    assert Path(result.uri).exists()


async def test_run_total_duration_is_sum_of_clips(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    clips = [
        _write_clip(tmp_path, "s01.mp4", 2.5),
        _write_clip(tmp_path, "s02.mp4", 3.0),
        _write_clip(tmp_path, "s03.mp4", 4.0),
    ]
    req = _request(tmp_path, clips=clips)

    with patch.object(adapter, "_run_ffmpeg", side_effect=_noop_ffmpeg):
        result = await adapter.run(req)

    assert result.duration_sec == pytest.approx(9.5)


async def test_run_creates_concat_and_caption_files(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    req = _request(tmp_path)

    with patch.object(adapter, "_run_ffmpeg", side_effect=_noop_ffmpeg):
        await adapter.run(req)

    assert (tmp_path / "output" / "concat.txt").exists()
    assert (tmp_path / "output" / "caption.txt").exists()


async def test_run_caption_file_contains_caption_text(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    req = _request(tmp_path, caption_text="Count to five with Pippa!")

    with patch.object(adapter, "_run_ffmpeg", side_effect=_noop_ffmpeg):
        await adapter.run(req)

    caption = (tmp_path / "output" / "caption.txt").read_text()
    assert "Count to five" in caption


async def test_run_raises_when_clip_missing(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    missing = VideoClip(uri=str(tmp_path / "ghost.mp4"), duration_sec=2.0, shot_id="s99")
    req = _request(tmp_path, clips=[missing])

    with pytest.raises(FileNotFoundError, match="lip_sync"):
        await adapter.run(req)


async def test_run_ffmpeg_command_received(tmp_path: Path) -> None:
    """Verify the ffmpeg command passed to _run_ffmpeg includes key flags."""
    adapter = _adapter(tmp_path)
    req = _request(tmp_path)
    received: list[list[str]] = []

    async def capture(cmd: list[str]) -> None:
        received.append(cmd)
        out = Path(cmd[-1])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"fake")

    with patch.object(adapter, "_run_ffmpeg", side_effect=capture):
        await adapter.run(req)

    assert received, "expected _run_ffmpeg to be called"
    cmd = received[0]
    assert "-filter_complex" in cmd
    assert "-shortest" in cmd
    assert "final.mp4" in cmd[-1]


# ------------------------------------------------------------------ estimate_cost

async def test_estimate_cost_is_zero(tmp_path: Path) -> None:
    cost = await _adapter(tmp_path).estimate_cost(_request(tmp_path))
    assert cost.amount == 0.0
