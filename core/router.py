from core.registry import AdapterRecord


def rank_adapters(records: list[AdapterRecord]) -> list[AdapterRecord]:
    enabled = [record for record in records if record.enabled and record.health_score > 0]
    return sorted(
        enabled,
        key=lambda record: (-record.priority, -record.quality_score, -record.health_score, record.adapter_name),
    )

