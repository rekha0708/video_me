from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from pydantic import BaseModel

from core.models.common import CostEstimate, HealthStatus

RequestT = TypeVar("RequestT", bound=BaseModel)
ResultT = TypeVar("ResultT", bound=BaseModel)


class Capability(ABC, Generic[RequestT, ResultT]):
    name: str
    version: str

    @abstractmethod
    async def health(self) -> HealthStatus:
        """Return the adapter's current health."""

    @abstractmethod
    async def estimate_cost(self, req: RequestT) -> CostEstimate:
        """Estimate cost before running a request."""

    @abstractmethod
    async def run(self, req: RequestT) -> ResultT:
        """Run the capability."""


class FetchMedia(Capability[BaseModel, BaseModel], ABC):
    name = "fetch_media"


class SeparateAudio(Capability[BaseModel, BaseModel], ABC):
    name = "separate_audio"


class Transcribe(Capability[BaseModel, BaseModel], ABC):
    name = "transcribe"


class AnalyzeContent(Capability[BaseModel, BaseModel], ABC):
    name = "analyze_content"


class AdaptScript(Capability[BaseModel, BaseModel], ABC):
    name = "adapt_script"


class PlanShots(Capability[BaseModel, BaseModel], ABC):
    name = "plan_shots"


class RenderCharacter(Capability[BaseModel, BaseModel], ABC):
    name = "render_character"


class SynthesizeVoice(Capability[BaseModel, BaseModel], ABC):
    name = "synthesize_voice"


class GenerateVideo(Capability[BaseModel, BaseModel], ABC):
    name = "generate_video"


class LipSync(Capability[BaseModel, BaseModel], ABC):
    name = "lip_sync"


class MixAudio(Capability[BaseModel, BaseModel], ABC):
    name = "mix_audio"


class AssembleVideo(Capability[BaseModel, BaseModel], ABC):
    name = "assemble_video"


class Critique(Capability[BaseModel, BaseModel], ABC):
    name = "critique"


class Publish(Capability[BaseModel, BaseModel], ABC):
    name = "publish"

