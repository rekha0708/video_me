import asyncio
import gc
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from core.capabilities.base import Transcribe
from core.models.capabilities import (
    TranscribeRequest,
    TranscribeResult,
    TranscriptSegment,
    WordTimestamp,
)
from core.models.common import CostEstimate, HealthStatus
from core.observability import log_event

if TYPE_CHECKING:
    from faster_whisper import WhisperModel as _WhisperModel

logger = logging.getLogger(__name__)

# Dedicated venv (python3 -m venv --system-site-packages) with demucs installed —
# see CLAUDE.md "Venv strategy". Optional: vocal isolation degrades gracefully to
# the raw mix if this venv or the subprocess call is unavailable.
_DEMUCS_PYTHON = Path("/workspace/.venv_demucs/bin/python")
_DEMUCS_MODEL = "htdemucs"
_DEMUCS_TIMEOUT_SEC = 600


class WhisperAdapter(Transcribe):
    """
    Transcription adapter using faster-whisper (CTranslate2 backend).

    Model is lazy-loaded on first run() call — init is cheap.
    Not thread-safe across concurrent run() calls; for Phase 1 single-job
    execution this is fine. Add a lock before enabling parallel jobs.

    Args:
        model_size:    faster-whisper model size (tiny/base/small/medium/large-v3).
        device:        "cpu" for local testing; "cuda" on rented GPU.
        compute_type:  "int8" for CPU; "float16" for CUDA GPU.
        beam_size:     Beam search width. 5 matches OpenAI Whisper default.
    """

    version = "1.0.0"

    def __init__(
        self,
        model_size: str = "medium",
        device: str = "cpu",
        compute_type: str = "int8",
        beam_size: int = 5,
        vad_filter: bool = False,
        language: str = "",
        download_root: str = "",
        local_files_only: bool = True,
        revision: str = "",
    ) -> None:
        self._model_size = model_size
        self._device = device
        self._compute_type = compute_type
        self._beam_size = beam_size
        self._vad_filter = vad_filter
        self._download_root = download_root.strip() or None
        self._local_files_only = local_files_only
        self._revision = revision.strip() or None
        normalized_language = language.strip().lower()
        self._language = None if normalized_language in ("", "auto") else normalized_language
        self._model: "_WhisperModel | None" = None

    async def health(self) -> HealthStatus:
        try:
            import faster_whisper  # noqa: F401
        except ImportError:
            return HealthStatus(
                status="down",
                reason="faster-whisper not installed. Run: pip install faster-whisper",
            )
        return HealthStatus(status="ok")

    async def estimate_cost(self, req: TranscribeRequest) -> CostEstimate:
        return CostEstimate(amount=0.0, notes="Local Whisper model; no per-call API cost.")

    async def run(self, req: TranscribeRequest) -> TranscribeResult:
        log_event(
            logger,
            "transcribe_started",
            audio_uri=req.audio_uri,
            model=self._model_size,
            isolate_vocals=req.isolate_vocals,
        )
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, self._transcribe, req.audio_uri, req.isolate_vocals
        )
        log_event(
            logger,
            "transcribe_completed",
            audio_uri=req.audio_uri,
            language=result.language,
            segment_count=len(result.segments),
        )
        return result

    # ------------------------------------------------------------------
    # Private helpers (mockable in unit tests)
    # ------------------------------------------------------------------

    def _ensure_model(self) -> "_WhisperModel":
        if self._model is None:
            from faster_whisper import WhisperModel
            log_event(
                logger,
                "whisper_model_loading",
                model_size=self._model_size,
                device=self._device,
                compute_type=self._compute_type,
            )
            self._model = WhisperModel(
                self._model_size,
                device=self._device,
                compute_type=self._compute_type,
                download_root=self._download_root,
                local_files_only=self._local_files_only,
                revision=self._revision,
            )
        return self._model

    async def unload(self) -> bool:
        """Drop the local Whisper model and ask CUDA to release cached memory."""
        if self._model is None:
            return True

        model = self._model
        self._model = None
        del model
        gc.collect()

        try:
            import torch
        except ImportError:
            torch = None  # type: ignore[assignment]

        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()

        log_event(logger, "whisper_model_unloaded", model_size=self._model_size)
        return True

    def _transcribe(self, audio_uri: str, isolate_vocals: bool = False) -> TranscribeResult:
        transcribe_uri = audio_uri
        if isolate_vocals:
            transcribe_uri = self._isolate_vocals(audio_uri)

        model = self._ensure_model()

        kwargs = {
            "beam_size": self._beam_size,
            "word_timestamps": True,
            "vad_filter": self._vad_filter,
        }
        if self._language:
            kwargs["language"] = self._language

        segments_iter, info = model.transcribe(transcribe_uri, **kwargs)

        segments: list[TranscriptSegment] = []
        full_text_parts: list[str] = []

        for seg in segments_iter:
            words = [
                WordTimestamp(word=w.word.strip(), start=w.start, end=w.end)
                for w in (seg.words or [])
                if w.word.strip()
            ]
            segments.append(
                TranscriptSegment(
                    text=seg.text.strip(),
                    start=seg.start,
                    end=seg.end,
                    words=words,
                )
            )
            full_text_parts.append(seg.text.strip())

        return TranscribeResult(
            segments=segments,
            language=info.language,
            full_text=" ".join(full_text_parts),
        )

    def _isolate_vocals(self, audio_uri: str) -> str:
        """Run Demucs source separation and return the vocals-stem path.

        Best-effort: falls back to the original mix (with a warning) if the
        demucs venv is missing, the subprocess fails, or times out — a bad
        separation should never block the whole transcribe stage.
        """
        if not _DEMUCS_PYTHON.exists():
            log_event(
                logger,
                "vocal_isolation_skipped",
                reason="demucs venv not found",
                expected_path=str(_DEMUCS_PYTHON),
            )
            return audio_uri

        audio_path = Path(audio_uri)
        out_dir = Path(tempfile.mkdtemp(prefix="_demucs_", dir=str(audio_path.parent)))
        track_stem = audio_path.stem
        vocals_path = out_dir / _DEMUCS_MODEL / track_stem / "vocals.wav"

        log_event(logger, "vocal_isolation_started", audio_uri=audio_uri)
        try:
            proc = subprocess.run(
                [
                    str(_DEMUCS_PYTHON), "-m", "demucs.separate",
                    "--two-stems=vocals", "-n", _DEMUCS_MODEL,
                    "-o", str(out_dir), str(audio_path),
                ],
                capture_output=True,
                text=True,
                timeout=_DEMUCS_TIMEOUT_SEC,
            )
        except (subprocess.TimeoutExpired, OSError) as exc:
            log_event(logger, "vocal_isolation_failed", reason=str(exc))
            return audio_uri

        if proc.returncode != 0 or not vocals_path.exists():
            log_event(
                logger,
                "vocal_isolation_failed",
                reason="demucs exited nonzero or output missing",
                returncode=proc.returncode,
                stderr=proc.stderr[-2000:] if proc.stderr else "",
            )
            return audio_uri

        log_event(logger, "vocal_isolation_completed", vocals_path=str(vocals_path))
        return str(vocals_path)
