from pathlib import Path
from typing import TypeVar

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
    postgres_dsn: str = "postgresql://video_me:video_me_dev@localhost:5432/video_me"
    s3_endpoint_url: str = "http://localhost:9000"
    s3_bucket: str = "video-me-artifacts"
    workflow_engine: str = "prefect"
    max_regenerations: int = 3


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
    cast_path: Path = Path("config/casts/pig_kids_placeholder.yaml"),
) -> AppConfig:
    return AppConfig(
        channel_profile=load_yaml_model(channel_path, ChannelProfile),
        cast=load_yaml_model(cast_path, Cast),
    )

