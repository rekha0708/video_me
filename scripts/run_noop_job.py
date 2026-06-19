import asyncio

from core.observability import configure_logging
from core.workflow import run_noop_job


async def main() -> None:
    configure_logging()
    job = await run_noop_job()
    print(job.model_dump_json(indent=2))


if __name__ == "__main__":
    asyncio.run(main())

