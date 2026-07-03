"""Pipeline Assistant chat service.

Streams LLM responses over SSE, with tool calling for live job state.
Supports any OpenAI-compatible endpoint via VIDEO_ME_CHAT_* env vars.
"""
from __future__ import annotations

import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from core.models.dashboard import DashboardJobStatus
from services.dashboard_repository import DashboardRepository

_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_job_status",
            "description": (
                "Get live status, phase, current stage, and error details for the current job."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_job_events",
            "description": (
                "Get recent pipeline events for the current job "
                "(logs, errors, stage transitions)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Number of recent events (max 50, default 20)",
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_all_jobs",
            "description": "List all pipeline jobs with their current status, phase, and stage.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_service_health",
            "description": "Check the health of the pipeline worker and services.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cancel_job",
            "description": (
                "Request cancellation of the current job. "
                "Only call after the user explicitly confirms."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "confirm": {
                        "type": "boolean",
                        "description": (
                            "Must be true. Call only when the user has explicitly agreed to cancel."
                        ),
                    }
                },
                "required": ["confirm"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_job_errors",
            "description": (
                "Return only ERROR-level events for the current job, with full payload including "
                "stack traces. Call this first when the user asks why a job failed."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Number of error events to return (max 20, default 10)",
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_service_log",
            "description": (
                "Read the last N lines of a service log file. Useful when a stage fails with a "
                "service error (e.g. ComfyUI HTTP 500, Fish S2 crash). "
                "Services: ollama, comfyui, fish_s2, wan, musetalk, a1111, chatterbox."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "service": {
                        "type": "string",
                        "enum": ["ollama", "comfyui", "fish_s2", "wan", "musetalk", "a1111", "chatterbox"],
                        "description": "Which service log to read.",
                    },
                    "lines": {
                        "type": "integer",
                        "description": "Number of trailing log lines to return (default 50, max 200).",
                    },
                },
                "required": ["service"],
            },
        },
    },
]

_TERMINAL_STATUSES = {"completed", "failed", "blocked", "cancelled"}


def _build_system_prompt(job_id: str, repo: DashboardRepository) -> str:
    job = repo.get_job(job_id)
    if not job:
        return (
            "You are a Pipeline Assistant for the video_me dashboard. "
            "No specific job context. Use list_all_jobs to see available jobs."
        )
    events = repo.list_events(job_id, limit=5)
    event_lines = "\n".join(
        f"  [{e.created_at.strftime('%H:%M:%S') if e.created_at else '?'}]"
        f" [{e.level.value.upper()}]"
        f" {f'{e.stage_name}: ' if e.stage_name else ''}{e.message}"
        for e in events[-5:]
    ) or "  (no events yet)"

    error_block = ""
    if job.terminal_error:
        error_block = f"\nTerminal error: {json.dumps(job.terminal_error)}"

    src = job.source_url
    if len(src) > 80:
        src = src[:80] + "..."

    return (
        "You are a Pipeline Assistant for the video_me dashboard.\n\n"
        "Current job:\n"
        f"  Job ID: {job.job_id}\n"
        f"  Status: {job.status.value}\n"
        f"  Phase: {job.phase}\n"
        f"  Current stage: {job.current_stage or 'none'}\n"
        f"  Source: {src}{error_block}\n\n"
        f"Recent events (last 5):\n{event_lines}\n\n"
        "You have tools to read live job state and take actions.\n"
        "When asked why a job failed: call get_job_errors() first — it returns full stack traces.\n"
        "When a stage failure mentions a service (ComfyUI, Fish S2, Ollama): call get_service_log() to see the raw service output.\n"
        "For cancel_job: always ask the user to confirm explicitly before calling it.\n"
        "Keep responses concise and reference actual event messages when diagnosing failures."
    )


async def _execute_tool(
    name: str,
    args: dict[str, Any],
    repo: DashboardRepository,
    job_id: str,
) -> dict[str, Any]:
    if name == "get_job_status":
        job = repo.get_job(job_id)
        if not job:
            return {"error": f"Job {job_id} not found"}
        return {
            "job_id": job.job_id,
            "status": job.status.value,
            "phase": job.phase,
            "current_stage": job.current_stage,
            "target_language": job.target_language,
            "source_url": job.source_url,
            "started_at": job.started_at.isoformat() if job.started_at else None,
            "completed_at": job.completed_at.isoformat() if job.completed_at else None,
            "terminal_error": job.terminal_error,
        }

    if name == "get_job_events":
        limit = min(int(args.get("limit", 20)), 50)
        events = repo.list_events(job_id, limit=limit)
        return {
            "events": [
                {
                    "event_type": e.event_type,
                    "level": e.level.value,
                    "stage_name": e.stage_name,
                    "message": e.message,
                    "payload": e.payload,
                    "created_at": e.created_at.isoformat() if e.created_at else None,
                }
                for e in events
            ]
        }

    if name == "list_all_jobs":
        jobs = repo.list_jobs(limit=20)
        return {
            "jobs": [
                {
                    "job_id": j.job_id,
                    "status": j.status.value,
                    "phase": j.phase,
                    "current_stage": j.current_stage,
                }
                for j in jobs
            ]
        }

    if name == "get_service_health":
        hb = repo.latest_worker_heartbeat()
        return {
            "worker": {
                "active": hb is not None,
                "worker_id": hb.worker_id if hb else None,
                "current_job_id": hb.current_job_id if hb else None,
                "last_heartbeat": (
                    hb.last_heartbeat_at.isoformat()
                    if hb and hb.last_heartbeat_at
                    else None
                ),
            }
        }

    if name == "cancel_job":
        if not args.get("confirm"):
            return {"error": "confirm must be true. Ask the user to confirm before calling."}
        job = repo.get_job(job_id)
        if not job:
            return {"error": f"Job {job_id} not found"}
        if job.status.value in _TERMINAL_STATUSES:
            return {"error": f"Job is already terminal: {job.status.value}"}
        repo.update_job_status(job_id, DashboardJobStatus.CANCEL_REQUESTED)
        repo.record_event(
            job_id,
            "cancel_requested",
            "Cancellation requested via Pipeline Assistant.",
        )
        return {"result": "Cancellation requested. Worker will stop within ~30 seconds."}

    if name == "get_job_errors":
        limit = min(int(args.get("limit", 10)), 20)
        events = repo.get_error_events(job_id, limit=limit)
        return {
            "error_events": [
                {
                    "event_type": e.event_type,
                    "stage_name": e.stage_name,
                    "message": e.message,
                    "payload": e.payload,
                    "created_at": e.created_at.isoformat() if e.created_at else None,
                }
                for e in events
            ]
        }

    if name == "get_service_log":
        service = args.get("service", "")
        n_lines = min(int(args.get("lines", 50)), 200)
        workspace = Path(
            __import__("os").environ.get("WORKSPACE", "/workspace")
        )
        log_path = workspace / "logs" / f"{service}.log"
        if not log_path.exists():
            return {"error": f"Log file not found: {log_path}"}
        try:
            text = log_path.read_text(errors="replace")
            tail = "\n".join(text.splitlines()[-n_lines:])
            return {"service": service, "log_path": str(log_path), "lines": tail}
        except OSError as exc:
            return {"error": f"Could not read {log_path}: {exc}"}

    return {"error": f"Unknown tool: {name}"}


async def chat_stream(
    job_id: str,
    user_message: str,
    repo: DashboardRepository,
    settings: Any,
) -> AsyncIterator[str]:
    """Async generator yielding SSE lines for one chat turn.

    SSE payload types:
      {"type": "token",     "content": "..."}   — streamed text fragment
      {"type": "tool_call", "name": "...", "result": {...}}   — tool execution
      {"type": "done"}                          — turn complete
      {"type": "error",    "message": "..."}   — LLM error
    """
    from openai import AsyncOpenAI

    client = AsyncOpenAI(
        base_url=settings.chat_base_url,
        api_key=settings.chat_api_key,
    )
    model = settings.chat_model

    repo.save_chat_message(job_id, "user", user_message)

    system_prompt = _build_system_prompt(job_id, repo)
    history = repo.get_chat_history(job_id, limit=20)
    # Exclude the message we just saved (last entry) — it becomes the explicit user turn.
    history_messages: list[dict[str, Any]] = [
        {"role": msg.role.value, "content": msg.content}
        for msg in history[:-1]
    ]
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        *history_messages,
        {"role": "user", "content": user_message},
    ]

    content_parts: list[str] = []
    tool_calls_data: dict[int, dict[str, str]] = {}

    try:
        stream = await client.chat.completions.create(
            model=model,
            messages=messages,
            tools=_TOOLS,
            tool_choice="auto",
            stream=True,
            extra_body={"think": False},
            max_tokens=2048,
        )
        async for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta

            if delta.content:
                content_parts.append(delta.content)
                yield f'data: {json.dumps({"type": "token", "content": delta.content})}\n\n'

            if delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = tc.index
                    if idx not in tool_calls_data:
                        tool_calls_data[idx] = {"id": tc.id or "", "name": "", "args": ""}
                    if tc.id:
                        tool_calls_data[idx]["id"] = tc.id
                    if tc.function:
                        if tc.function.name:
                            tool_calls_data[idx]["name"] = tc.function.name
                        if tc.function.arguments:
                            tool_calls_data[idx]["args"] += tc.function.arguments

    except Exception as exc:
        yield f'data: {json.dumps({"type": "error", "message": str(exc)})}\n\n'
        yield f'data: {json.dumps({"type": "done"})}\n\n'
        return

    first_content = "".join(content_parts)

    if tool_calls_data:
        tool_result_messages: list[dict[str, Any]] = []
        assistant_tool_calls: list[dict[str, Any]] = []

        for idx in sorted(tool_calls_data.keys()):
            tc_data = tool_calls_data[idx]
            name = tc_data["name"]
            try:
                args: dict[str, Any] = json.loads(tc_data["args"]) if tc_data["args"] else {}
            except json.JSONDecodeError:
                args = {}

            result = await _execute_tool(name, args, repo, job_id)
            yield f'data: {json.dumps({"type": "tool_call", "name": name, "result": result})}\n\n'

            assistant_tool_calls.append({
                "id": tc_data["id"],
                "type": "function",
                "function": {"name": name, "arguments": tc_data["args"]},
            })
            tool_result_messages.append({
                "role": "tool",
                "tool_call_id": tc_data["id"],
                "content": json.dumps(result, default=str),
            })

        assistant_msg: dict[str, Any] = {
            "role": "assistant",
            "tool_calls": assistant_tool_calls,
        }
        if first_content:
            assistant_msg["content"] = first_content

        messages2 = messages + [assistant_msg] + tool_result_messages
        final_parts: list[str] = []

        try:
            stream2 = await client.chat.completions.create(
                model=model,
                messages=messages2,
                stream=True,
                extra_body={"think": False},
                max_tokens=2048,
            )
            async for chunk in stream2:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if delta.content:
                    final_parts.append(delta.content)
                    yield f'data: {json.dumps({"type": "token", "content": delta.content})}\n\n'
        except Exception as exc:
            yield f'data: {json.dumps({"type": "error", "message": str(exc)})}\n\n'

        final_content = "".join(final_parts)
        repo.save_chat_message(
            job_id, "assistant", final_content or first_content or "(tool call)"
        )
    else:
        repo.save_chat_message(job_id, "assistant", first_content)

    yield f'data: {json.dumps({"type": "done"})}\n\n'
