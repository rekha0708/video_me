# video_me

`video_me` is the orchestration repo for the Synthetic Kids' Educational Channel project.

The current implementation is Phase 0: contracts, data models, config loading, local storage/job
recording, structured logs, and a no-op workflow. See `BUILD_PROGRESS.md` for the step-by-step
implementation journal and rationale.

## Local Phase 0 Smoke Run

Install dependencies, then run:

```bash
python -m scripts.run_noop_job
```

The no-op workflow writes artifacts under `.local/artifacts/` and records job/stage state in
`.local/video_me.db`.

## Local Services

```bash
docker compose up
```

This starts local PostgreSQL and MinIO for later Track D integration. The current no-op workflow uses
local filesystem/SQLite defaults so it can run without cloud credentials.

