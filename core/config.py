from pathlib import Path
from typing import Literal, TypeVar

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from core.models.profile import Cast, ChannelProfile

ModelT = TypeVar("ModelT", bound=BaseModel)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="VIDEO_ME_", env_file=".env", extra="ignore")

    app_name: str = "video_me"
    environment: str = "local"
    data_dir: Path = Path(".local")
    artifact_dir: Path = Path(".local/artifacts")
    sqlite_path: Path = Path(".local/video_me.db")
    job_store: Literal["sqlite", "postgres"] = "sqlite"
    artifact_store: Literal["local", "s3"] = "local"
    postgres_dsn: str = "postgresql://video_me:video_me_dev@localhost:5432/video_me"
    s3_endpoint_url: str = "http://localhost:9000"
    s3_bucket: str = "video-me-artifacts"
    s3_access_key_id: str = "video_me"
    s3_secret_access_key: str = "video_me_dev_password"
    s3_region: str = "us-east-1"
    workflow_engine: str = "asyncio"
    max_regenerations: int = 3
    lora_dir: Path = Path("loras")
    voice_dir: Path = Path("voices")
    review_dir: Path = Path("review")
    llm_model: str = "qwen2.5:7b"
    llm_base_url: str = "http://localhost:11434/v1"
    llm_api_key: str = "ollama"
    critique_model: str = "llava:7b"
    critique_base_url: str = "http://localhost:11434/v1"
    critique_api_key: str = "ollama"
    sd_base_url: str = "http://localhost:7860"
    tts_base_url: str = "http://localhost:8020"
    wan_base_url: str = "http://localhost:8030"
    lipsync_base_url: str = "http://localhost:8040"
    whisper_model_size: str = "medium"
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"
    ffmpeg_bin: str = "ffmpeg"
    ffprobe_bin: str = "ffprobe"
    render_allow_placeholder_lora: bool = False


class AppConfig(BaseModel):
    settings: Settings = Field(default_factory=Settings)
    channel_profile: ChannelProfile
    cast: Cast


def load_yaml_model(path: Path, model: type[ModelT]) -> ModelT:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    return model.model_validate(payload)


def load_app_config(
    channel_path: Path = Path("config/channels/education_kids.yaml"),
    cast_path: Path = Path("config/casts/kids_duo.yaml"),
) -> AppConfig:
    return AppConfig(
        channel_profile=load_yaml_model(channel_path, ChannelProfile),
        cast=load_yaml_model(cast_path, Cast),
    )
