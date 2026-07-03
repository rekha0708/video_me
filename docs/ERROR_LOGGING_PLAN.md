# Plan: Improved Error Logging and Failure Tracking

## Context

Pipeline failures are currently hard to diagnose because:
1. Full stack traces are lost to stdout — only `{"code": "ExceptionClassName", "message": "str(exc)"}` is stored in the DB.
2. Stage-level failures don't get their own event — only a generic `job_failed` event is recorded, with no indication of *which stage* failed or *why* at a structured level.
3. The chatbot's `get_job_events()` tool omits the `payload` field, so it can't see error details even when they're stored.
4. No way for the chatbot (or a human) to read adapter service logs (ComfyUI, Fish S2, yt-dlp stderr) that contain the real root cause.
5. `log_event()` always emits at INFO level — errors don't stand out in stdout.

Goal: make every failure fully traceable — in the DB, in the dashboard UI, and via the chatbot — without changing the existing happy-path behavior.

---

## Files to Change (7 files)

### 1. `core/observability.py` — add `level` param to `log_event()`

Add an optional `level: int = logging.INFO` parameter. Error-path callers pass `logging.ERROR`. No change for existing callers.

```python
def log_event(logger, event_name, *,  level=logging.INFO, **fields):
    ...
    logger.log(level, json.dumps({...}))
```

---

### 2. `core/workflow.py` — add `error_hook` to `RunOptions`

`RunOptions` already has `stage_hook: Callable[[str, str], None] | None`. Add alongside it:

```python
error_hook: Callable[[str, Exception], Awaitable[None]] | None = None
```

The error hook receives `(stage_name, exc)` and is called before re-raising from the executor. The worker wires it up; the CLI path leaves it None.

---

### 3. `core/executor.py` — call `error_hook` + log at ERROR level on failure

In `run_stage()`, the current failure block catches the exception and re-raises after calling `stage_hook(..., "stage_failed")`. Enhance it:

```python
except Exception as exc:
    log_event(logger, "stage_failed", level=logging.ERROR,
              job_id=job.job_id, stage=stage_name, adapter=adapter_name,
              error=type(exc).__name__, message=str(exc))
    if options.stage_hook:
        options.stage_hook(stage_name, "stage_failed")
    if options.error_hook:
        await options.error_hook(stage_name, exc)   # ← NEW
    raise
```

---

### 4. `services/dashboard_worker.py` — three changes

**A. Full traceback in `_handle_failure()`**

```python
import traceback as _traceback

def _handle_failure(self, job_id, queue_id, exc):
    error = {
        "code": type(exc).__name__,
        "message": str(exc),
        "traceback": _traceback.format_exc(),   # ← NEW
    }
    ...
```

**B. New `_make_error_hook()` method** — records a `stage_failed` event with structured error details before the generic `job_failed`:

```python
def _make_error_hook(self, job_id: str):
    async def hook(stage_name: str, exc: Exception) -> None:
        import traceback as _tb
        self.repo.record_event(
            job_id,
            "stage_failed",
            f"{type(exc).__name__}: {exc}",
            level=DashboardEventLevel.ERROR,
            stage_name=stage_name,
            payload={
                "code": type(exc).__name__,
                "message": str(exc),
                "traceback": _tb.format_exc(),
            },
        )
    return hook
```

**C. Wire `error_hook` into `RunOptions`** — in the `_run_pipeline()` method where `RunOptions` is constructed:

```python
options = RunOptions(
    phase=phase,
    resume=resume,
    stage_hook=self._make_stage_hook(job_id),
    error_hook=self._make_error_hook(job_id),   # ← NEW
)
```

---

### 5. `services/dashboard_repository.py` — add `get_error_events()`

New method alongside `get_events()`:

```python
def get_error_events(self, job_id: str, limit: int = 20) -> list[DashboardEvent]:
    """Return only ERROR-level events for a job, most recent first."""
    ...  # same query as get_events() with WHERE level='error' added
```

Used by the new chatbot tool.

---

### 6. `services/chat_service.py` — three changes

**A. Add `payload` to `get_job_events()` return**

In the tool's return mapping, add:
```python
"payload": ev.payload or {},
```

**B. Add `get_job_errors()` tool** — dedicated error diagnosis tool:

```python
{
    "name": "get_job_errors",
    "description": "Return only ERROR-level events for the current job, with full payload including stack traces. Use this first when asked why a job failed.",
    "parameters": {"type": "object", "properties": {
        "limit": {"type": "integer", "default": 10}
    }}
}
```

Implementation calls `repo.get_error_events(job_id, limit)` and returns events with full payload including traceback.

**C. Add `get_service_log()` tool** — read tail of a service's log file:

```python
{
    "name": "get_service_log",
    "description": "Read the last N lines of a service log file (ollama, comfyui, fish_s2, wan, musetalk, a1111). Returns raw log output — useful when a stage fails with a service error.",
    "parameters": {"type": "object", "properties": {
        "service": {"type": "string", "enum": ["ollama","comfyui","fish_s2","wan","musetalk","a1111","chatterbox"]},
        "lines": {"type": "integer", "default": 50}
    }, "required": ["service"]}
}
```

Implementation: reads `/workspace/logs/{service}.log` (the path `start_services.sh` uses), returns last N lines. If not found, returns a helpful message.

**D. Update system prompt** to mention the two new tools and instruct the model to call `get_job_errors()` as the first step when a user asks why a job failed.

---

### 7. `services/templates/job_detail.html` + `services/static/app.css` — expandable payload rows

Events with non-empty payloads get a `▶` toggle. Clicking expands a `<pre>` showing the payload JSON (pretty-printed). Error events automatically start expanded.

```html
{% for ev in detail.events %}
<div class="ev-row{% if ev.payload %} ev-has-detail{% endif %}"
     {% if ev.payload %}onclick="toggleEvDetail(this)"{% endif %}>
  <span class="ev-time">...</span>
  <span class="ev-level ev-level-{{ ev.level.value }}">...</span>
  <span class="ev-msg">{{ ev.message }}</span>
  {% if ev.stage_name %}<span class="ev-stage">{{ ev.stage_name }}</span>{% endif %}
  {% if ev.payload %}<span class="ev-toggle">▶</span>{% endif %}
</div>
{% if ev.payload %}
<div class="ev-detail{% if ev.level.value == 'error' %} ev-detail-open{% endif %}">
  <pre>{{ ev.payload | tojson(indent=2) }}</pre>
</div>
{% endif %}
{% endfor %}
```

CSS additions (in `app.css`):
```css
.ev-has-detail { cursor: pointer; }
.ev-detail     { display: none; padding: 6px 12px; background: var(--surface2); }
.ev-detail-open{ display: block; }
.ev-toggle     { margin-left: auto; font-size: 10px; color: var(--muted); }
```

JS toggle (inline in `job_detail.html`):
```javascript
function toggleEvDetail(row) {
  const detail = row.nextElementSibling;
  const open = detail.classList.toggle('ev-detail-open');
  row.querySelector('.ev-toggle').textContent = open ? '▼' : '▶';
}
```

---

## What does NOT change

- Happy-path behavior — no new DB columns, no schema migration needed (payload is already stored; traceback goes into the existing `payload_json` field of events, and `terminal_error_json` for the job row)
- Existing `stage_hook` interface — `error_hook` is additive
- Test suite — existing mocks don't call `error_hook`; new hook is None in all non-worker callers

---

## Verification

1. Run a job with a deliberate failure (e.g., point ComfyUI URL at a dead port)
2. Check dashboard events feed → should see a `stage_failed` event (red, with stage name) **before** `job_failed`, expandable to show traceback
3. Open chat drawer → ask "why did this job fail?" → chatbot calls `get_job_errors()` and cites the traceback
4. Ask chatbot "show me the comfyui log" → calls `get_service_log("comfyui")` → returns last 50 lines
5. Check `job.terminal_error` via `GET /api/jobs/{id}` → should now include `"traceback"` field
6. Run `python -m pytest -q` — all 333 tests pass (no interface breaks)
