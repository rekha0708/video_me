from pydantic import BaseModel, Field


class AdapterRecord(BaseModel):
    capability_name: str
    adapter_name: str
    version: str
    enabled: bool = True
    priority: int = 100
    resource_profile: dict[str, str | int | float] = Field(default_factory=dict)
    cost_profile: dict[str, str | int | float] = Field(default_factory=dict)
    quality_score: float = 0.5
    health_score: float = 1.0


class Registry:
    def __init__(self) -> None:
        self._records: dict[str, list[AdapterRecord]] = {}

    def register(self, record: AdapterRecord) -> None:
        self._records.setdefault(record.capability_name, []).append(record)

    def list(self, capability_name: str) -> list[AdapterRecord]:
        return list(self._records.get(capability_name, []))

