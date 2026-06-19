from abc import ABC, abstractmethod

from core.models.capabilities import (
    AdaptScriptRequest,
    AnalyzeRequest,
    AssembleRequest,
    AudioTrack,
    CritiqueRequest,
    CritiqueResult,
    FetchMediaRequest,
    FetchMediaResult,
    FinalVideo,
    ImageSet,
    LipSyncRequest,
    MixAudioRequest,
    PlanShotsRequest,
    PublishRequest,
    PublishResult,
    RenderCharacterRequest,
    SeparateAudioRequest,
    SeparateAudioResult,
    TranscribeRequest,
    TranscribeResult,
    VideoClip,
    VideoRequest,
    VoiceRequest,
)
from core.models.common import CostEstimate, HealthStatus
from core.models.content import ContentMetadata, Script, Storyboard


class Capability[RequestT, ResultT](ABC):
    name: str
    version: str

    @abstractmethod
    async def health(self) -> HealthStatus: ...

    @abstractmethod
    async def estimate_cost(self, req: RequestT) -> CostEstimate: ...

    @abstractmethod
    async def run(self, req: RequestT) -> ResultT: ...


class FetchMedia(Capability[FetchMediaRequest, FetchMediaResult], ABC):
    name = "fetch_media"


class SeparateAudio(Capability[SeparateAudioRequest, SeparateAudioResult], ABC):
    name = "separate_audio"


class Transcribe(Capability[TranscribeRequest, TranscribeResult], ABC):
    name = "transcribe"


class AnalyzeContent(Capability[AnalyzeRequest, ContentMetadata], ABC):
    name = "analyze_content"


class AdaptScript(Capability[AdaptScriptRequest, Script], ABC):
    name = "adapt_script"


class PlanShots(Capability[PlanShotsRequest, Storyboard], ABC):
    name = "plan_shots"


class RenderCharacter(Capability[RenderCharacterRequest, ImageSet], ABC):
    name = "render_character"


class SynthesizeVoice(Capability[VoiceRequest, AudioTrack], ABC):
    name = "synthesize_voice"


class GenerateVideo(Capability[VideoRequest, VideoClip], ABC):
    name = "generate_video"


class LipSync(Capability[LipSyncRequest, VideoClip], ABC):
    name = "lip_sync"


class MixAudio(Capability[MixAudioRequest, AudioTrack], ABC):
    name = "mix_audio"


class AssembleVideo(Capability[AssembleRequest, FinalVideo], ABC):
    name = "assemble_video"


class Critique(Capability[CritiqueRequest, CritiqueResult], ABC):
    name = "critique"


class Publish(Capability[PublishRequest, PublishResult], ABC):
    name = "publish"
