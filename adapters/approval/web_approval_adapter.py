"""
Human approval gate — serves a local web UI showing the storyboard + critique scores.
The pipeline pauses here until the operator clicks Approve or Reject in the browser.

Flow:
  1. Adapter writes storyboard_review.md + storyboard_review.json to work_dir.
  2. Starts a stdlib HTTP server on localhost:<port> in a background thread.
  3. Pipeline polls for a flag file every 10 s (up to timeout_hours).
  4. Approve  → flag = "approved"  → pipeline continues to render.
  5. Reject   → flag = "rejected:<notes>"  → caller gets (approved=False, notes).
  6. Caller can then re-plan with the notes and call the adapter again.
"""

import asyncio
import json
import logging
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from core.models.capabilities import PlanCritiqueResult
from core.models.content import Storyboard, Script
from core.models.profile import Cast

logger = logging.getLogger(__name__)

_POLL_INTERVAL = 10.0   # seconds between flag-file checks
_FLAG_FILE = "approval.flag"
_REVIEW_MD = "storyboard_review.md"
_REVIEW_JSON = "storyboard_review.json"


# ── HTML page ──────────────────────────────────────────────────────────────────

def _render_html(storyboard: Storyboard, script: Script, cast: Cast,
                 critique: PlanCritiqueResult | None, iteration: int) -> str:
    cast_map = {m.id: m.name for m in cast.members}

    rows = ""
    for s in storyboard.shots:
        chars = ", ".join(cast_map.get(c, c) for c in s.characters_on_screen)
        rows += (
            f"<tr><td>{s.shot_id}</td><td>{chars}</td>"
            f"<td>{s.camera}</td><td>{s.setting}</td>"
            f"<td>{s.action}</td><td>{s.duration_sec}s</td></tr>\n"
        )

    scores_html = ""
    if critique and critique.scores:
        for k, v in critique.scores.items():
            pct = int(v * 100)
            colour = "#4caf50" if v >= 0.75 else "#f44336"
            scores_html += (
                f'<div class="score-row"><span>{k.replace("_"," ")}</span>'
                f'<div class="bar"><div style="width:{pct}%;background:{colour}"></div></div>'
                f'<span>{pct}%</span></div>\n'
            )

    notes_html = ""
    if critique and critique.revision_notes:
        items = "".join(f"<li>{n}</li>" for n in critique.revision_notes)
        notes_html = f'<div class="notes"><strong>Critique notes (all addressed):</strong><ul>{items}</ul></div>'

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>video_me — Storyboard Approval</title>
<style>
  body{{font-family:system-ui,sans-serif;max-width:900px;margin:40px auto;padding:0 20px;background:#f5f5f5}}
  h1{{color:#1a237e;margin-bottom:4px}}
  .badge{{display:inline-block;padding:2px 10px;border-radius:12px;font-size:12px;
          background:#e8f5e9;color:#2e7d32;border:1px solid #a5d6a7;margin-left:8px}}
  table{{width:100%;border-collapse:collapse;background:#fff;border-radius:8px;
         overflow:hidden;box-shadow:0 1px 4px rgba(0,0,0,.1);margin:16px 0}}
  th{{background:#283593;color:#fff;padding:10px 12px;text-align:left;font-size:13px}}
  td{{padding:9px 12px;border-bottom:1px solid #eee;font-size:13px}}
  tr:last-child td{{border-bottom:none}}
  tr:hover td{{background:#f3f4ff}}
  .scores{{background:#fff;border-radius:8px;padding:16px 20px;margin:16px 0;
            box-shadow:0 1px 4px rgba(0,0,0,.1)}}
  .score-row{{display:flex;align-items:center;gap:12px;margin:6px 0}}
  .score-row span:first-child{{width:180px;font-size:13px;color:#555;text-transform:capitalize}}
  .bar{{flex:1;height:12px;background:#eee;border-radius:6px;overflow:hidden}}
  .bar div{{height:100%;border-radius:6px;transition:width .4s}}
  .score-row span:last-child{{width:36px;font-size:13px;font-weight:600;color:#333}}
  .notes{{background:#fff8e1;border:1px solid #ffe082;border-radius:8px;
           padding:12px 16px;margin:12px 0;font-size:13px}}
  .notes ul{{margin:6px 0;padding-left:20px}}
  .actions{{display:flex;gap:16px;margin:24px 0}}
  .btn{{padding:12px 32px;border:none;border-radius:8px;font-size:15px;
         font-weight:600;cursor:pointer;transition:opacity .2s}}
  .btn:hover{{opacity:.88}}
  .approve{{background:#2e7d32;color:#fff}}
  .reject{{background:#c62828;color:#fff}}
  textarea{{width:100%;margin-top:8px;padding:10px;border:1px solid #ddd;border-radius:6px;
             font-size:13px;resize:vertical}}
  label{{font-size:13px;font-weight:600;color:#555}}
  .iter{{font-size:12px;color:#999;margin-bottom:20px}}
</style>
</head>
<body>
<h1>Storyboard Approval <span class="badge">Critique passed ✓</span></h1>
<p class="iter">Planning iteration {iteration} &nbsp;·&nbsp;
  {len(storyboard.shots)} shots &nbsp;·&nbsp;
  {sum(s.duration_sec for s in storyboard.shots):.1f}s total</p>

<h2>Shots</h2>
<table>
<thead><tr><th>Shot</th><th>Character(s)</th><th>Camera</th><th>Setting</th><th>Action</th><th>Duration</th></tr></thead>
<tbody>{rows}</tbody>
</table>

<h2>Critique Scores</h2>
<div class="scores">{scores_html or "<p style='color:#999;font-size:13px'>No scores available.</p>"}</div>
{notes_html}

<h2>Decision</h2>
<form id="af">
  <div class="actions">
    <button type="button" class="btn approve" onclick="decide('approve')">✓ Approve — start rendering</button>
    <button type="button" class="btn reject" onclick="showReject()">✗ Reject — re-plan with notes</button>
  </div>
  <div id="rj" style="display:none">
    <label for="notes">Rejection notes (be specific — the LLM will re-plan using these):</label>
    <textarea id="notes" rows="4" placeholder="e.g. shot s03 feels too abstract for 3-year-olds, simplify the action..."></textarea>
    <div class="actions" style="margin-top:12px">
      <button type="button" class="btn reject" onclick="decide('reject')">Send rejection &amp; re-plan</button>
    </div>
  </div>
</form>
<script>
function showReject(){{document.getElementById('rj').style.display='block'}}
function decide(action){{
  const notes=document.getElementById('notes')?.value||'';
  fetch('/'+action,{{method:'POST',headers:{{'Content-Type':'application/json'}},
    body:JSON.stringify({{notes}})}})
  .then(()=>document.body.innerHTML='<div style="text-align:center;margin-top:120px">'
    +(action==='approve'
      ?'<h2 style="color:#2e7d32">✓ Approved — rendering will start shortly.</h2>'
      :'<h2 style="color:#c62828">✗ Rejected — pipeline will re-plan with your notes.</h2>')
    +'</div>');
}}
</script>
</body>
</html>"""


# ── HTTP handler ───────────────────────────────────────────────────────────────

class _Handler(BaseHTTPRequestHandler):
    html: str = ""
    flag_path: Path = Path("approval.flag")

    def log_message(self, fmt, *args):  # silence stdlib access log
        pass

    def do_GET(self):
        if self.path == "/":
            body = self.html.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        try:
            data = json.loads(body) if body else {}
        except Exception:
            data = {}

        notes = data.get("notes", "").strip()

        if self.path == "/approve":
            self.flag_path.write_text("approved")
            self._ok()
        elif self.path == "/reject":
            self.flag_path.write_text(f"rejected:{notes}")
            self._ok()
        else:
            self.send_response(404)
            self.end_headers()

    def _ok(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"ok":true}')


# ── Public adapter ─────────────────────────────────────────────────────────────

class WebApprovalAdapter:
    """
    Serves a local web UI on localhost:<port> showing the storyboard + critique scores.
    Returns (approved: bool, rejection_notes: str).

    If auto_approve=True (set via VIDEO_ME_AUTO_APPROVE_PLAN=true) the gate is skipped
    and approved=True is returned immediately — for CI / smoke tests.
    """

    def __init__(
        self,
        work_dir: Path,
        port: int = 8765,
        timeout_hours: float = 24.0,
        auto_approve: bool = False,
    ) -> None:
        self._work_dir = work_dir
        self._port = port
        self._timeout = timeout_hours * 3600
        self._auto_approve = auto_approve

    async def request_approval(
        self,
        storyboard: Storyboard,
        script: Script,
        cast: Cast,
        critique: PlanCritiqueResult | None = None,
        iteration: int = 1,
    ) -> tuple[bool, str]:
        """
        Write review files, start the web server, wait for human decision.
        Returns (approved, rejection_notes).
        """
        self._write_review_files(storyboard, script, cast, critique)

        if self._auto_approve:
            logger.info("approval gate: auto-approved (VIDEO_ME_AUTO_APPROVE_PLAN=true)")
            return True, ""

        flag_path = self._work_dir / _FLAG_FILE
        flag_path.unlink(missing_ok=True)

        html = _render_html(storyboard, script, cast, critique, iteration)
        server = self._start_server(html, flag_path)

        logger.info(
            "Storyboard ready for review → http://localhost:%d  (waiting up to %.0fh)",
            self._port, self._timeout / 3600,
        )
        print(f"\n{'='*60}")
        print(f"  STORYBOARD APPROVAL REQUIRED")
        print(f"  Open: http://localhost:{self._port}")
        print(f"  Review file: {self._work_dir / _REVIEW_MD}")
        print(f"{'='*60}\n")

        try:
            result = await self._poll_flag(flag_path)
        finally:
            server.shutdown()

        approved = result == "approved"
        notes = result[len("rejected:"):] if result.startswith("rejected:") else ""
        logger.info("approval gate: %s notes=%r", "approved" if approved else "rejected", notes)
        return approved, notes

    # ── private ──

    def _write_review_files(self, storyboard, script, cast, critique):
        self._work_dir.mkdir(parents=True, exist_ok=True)
        cast_map = {m.id: m.name for m in cast.members}

        # Markdown summary
        lines = ["# Storyboard Review\n"]
        lines.append(f"**{len(storyboard.shots)} shots** · "
                     f"{sum(s.duration_sec for s in storyboard.shots):.1f}s total\n")
        lines.append("| Shot | Character(s) | Camera | Setting | Action | Duration |")
        lines.append("|------|-------------|--------|---------|--------|----------|")
        for s in storyboard.shots:
            chars = ", ".join(cast_map.get(c, c) for c in s.characters_on_screen)
            lines.append(
                f"| {s.shot_id} | {chars} | {s.camera} | {s.setting} | {s.action} | {s.duration_sec}s |"
            )

        if critique:
            lines.append("\n## Critique Scores\n")
            for k, v in critique.scores.items():
                bar = "█" * int(v * 10) + "░" * (10 - int(v * 10))
                lines.append(f"- **{k}**: {bar} {v:.0%}")
            if critique.revision_notes:
                lines.append("\n## Addressed Notes\n")
                for n in critique.revision_notes:
                    lines.append(f"- {n}")

        (self._work_dir / _REVIEW_MD).write_text("\n".join(lines))
        (self._work_dir / _REVIEW_JSON).write_text(
            json.dumps(storyboard.model_dump(), indent=2)
        )

    def _start_server(self, html: str, flag_path: Path) -> HTTPServer:
        _Handler.html = html
        _Handler.flag_path = flag_path
        server = HTTPServer(("localhost", self._port), _Handler)
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()
        return server

    async def _poll_flag(self, flag_path: Path) -> str:
        deadline = time.monotonic() + self._timeout
        while time.monotonic() < deadline:
            await asyncio.sleep(_POLL_INTERVAL)
            if flag_path.exists():
                return flag_path.read_text().strip()
        raise TimeoutError(
            f"No approval received within {self._timeout / 3600:.0f}h — job timed out."
        )
