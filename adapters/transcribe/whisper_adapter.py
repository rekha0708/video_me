import asyncio
import logging
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
    ) -> None:
        self._model_size = model_size
        self._device = device
        self._compute_type = compute_type
        self._beam_size = beam_size
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
        log_event(logger, "transcribe_started", audio_uri=req.audio_uri, model=self._model_size)
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, self._transcribe, req.audio_uri)
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
            )
        return self._model

    def _transcribe(self, audio_uri: str) -> TranscribeResult:
        model = self._ensure_model()

        segments_iter, info = model.transcribe(
            audio_uri,
            beam_size=self._beam_size,
            word_timestamps=True,
            vad_filter=True,  # skip silence — faster and more accurate
        )

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
