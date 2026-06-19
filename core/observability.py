import json
import logging
from typing import Any


def configure_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(level=level, format="%(message)s")


def log_event(
    logger: logging.Logger,
    event: str,
    *,
    job_id: str | None = None,
    stage: str | None = None,
    adapter: str | None = None,
    **fields: Any,
) -> None:
    payload = {
        "event": event,
        "job_id": job_id,
        "stage": stage,
        "adapter": adapter,
        **fields,
    }
    logger.info(json.dumps({key: value for key, value in payload.items() if value is not None}, default=str))

