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

To run the same no-op workflow against local PostgreSQL and MinIO, start the services and set the
service-backed stores:

```bash
docker compose up -d
VIDEO_ME_JOB_STORE=postgres VIDEO_ME_ARTIFACT_STORE=s3 python -m scripts.run_noop_job
```

## Local Services

```bash
docker compose up
```

This starts local PostgreSQL and MinIO for later Track D integration. The current no-op workflow uses
local filesystem/SQLite defaults so it can run without cloud credentials unless the `VIDEO_ME_*`
store settings are enabled.

Default local service endpoints:

- PostgreSQL: `postgresql://video_me:video_me_dev@localhost:5432/video_me`
- MinIO API: `http://localhost:9000`
- MinIO console: `http://localhost:9001`
- MinIO credentials: `video_me` / `video_me_dev_password`
