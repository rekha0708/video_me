"""
Image approval gate — serves a grid web UI showing all shots' candidate images.

The pipeline pauses here after all shots are rendered and critiqued.
The operator can confirm the VLM pick or override to a different candidate for any shot.
Overrides are written back to the critique_feedback.jsonl log (self-learning).

Flow:
  1. Adapter builds an HTML grid: one row per shot, showing all N candidate images,
     with the VLM pick pre-selected and radio buttons to switch.
  2. Starts a stdlib HTTP server on localhost:<port> in a background thread.
     Images are served as static files via GET /img/<b64-encoded-path>.
  3. Pipeline polls for a flag file every 10 s (up to timeout_hours).
  4. Approve → flag = "approved:<json-overrides>" → caller gets approved URIs list.
  5. auto_approve=True returns immediately with the VLM picks (CI bypass).
"""

import asyncio
import base64
import json
import logging
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from core.models.capabilities import ImageApprovalRequest, ImageApprovalResult, ImageCritiqueResult
from core.models.content import Shot

logger = logging.getLogger(__name__)

_POLL_INTERVAL = 10.0
_FLAG_FILE = "image_approval.flag"


# ── HTML renderer ──────────────────────────────────────────────────────────────

def _render_html(req: ImageApprovalRequest) -> str:
    rows = ""
    for shot, critique in zip(req.shots, req.critique_results):
        shot_id = shot.shot_id if hasattr(shot, "shot_id") else str(shot)
        setting = getattr(shot, "setting", "")
        action = getattr(shot, "action", "")

        imgs_html = ""
        for i, uri in enumerate(critique.candidate_uris if hasattr(critique, "candidate_uris") else [critique.winner_uri]):
            checked = "checked" if i == critique.winner_index else ""
            # Encode path as URL-safe base64 for the image src route.
            path_b64 = base64.urlsafe_b64encode(uri.encode()).decode()
            score_tip = ""
            if i < len(critique.candidate_scores):
                cs = critique.candidate_scores[i]
                avg = sum(cs.scores.values()) / len(cs.scores) if cs.scores else 0
                score_tip = f"avg {avg:.0%}"
            imgs_html += f"""
<label class="cand {'winner' if i == critique.winner_index else ''}">
  <input type="radio" name="pick_{shot_id}" value="{i}" {checked}>
  <img src="/img/{path_b64}" alt="candidate {i}" loading="lazy">
  <span class="idx">#{i} {score_tip}</span>
</label>"""

        rows += f"""
<tr>
  <td class="meta">
    <strong>{shot_id}</strong><br>
    <span>{setting}</span><br>
    <em>{action[:60]}{'…' if len(action) > 60 else ''}</em>
  </td>
  <td class="cands">{imgs_html}</td>
</tr>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>video_me — Image Approval</title>
<style>
  body{{font-family:system-ui,sans-serif;max-width:1100px;margin:40px auto;padding:0 20px;
       background:#f5f5f5;font-size:13px}}
  h1{{color:#1a237e;margin-bottom:4px}}
  .sub{{color:#999;font-size:11px;margin-bottom:20px}}
  table{{width:100%;border-collapse:collapse;background:#fff;border-radius:10px;
         overflow:hidden;box-shadow:0 2px 8px rgba(0,0,0,.10)}}
  th{{background:#283593;color:#fff;padding:10px 14px;text-align:left;font-size:12px}}
  td{{padding:12px 14px;border-bottom:1px solid #eee;vertical-align:top}}
  tr:last-child td{{border-bottom:none}}
  td.meta{{width:180px;color:#555;line-height:1.5}}
  td.meta strong{{color:#1a237e;font-size:13px}}
  td.meta em{{color:#888}}
  td.cands{{display:flex;gap:10px;flex-wrap:wrap;align-items:flex-start}}
  label.cand{{display:flex;flex-direction:column;align-items:center;gap:4px;
              cursor:pointer;padding:6px;border-radius:8px;border:2px solid transparent;
              transition:border-color .2s}}
  label.cand:hover{{border-color:#7986cb}}
  label.cand.winner{{border-color:#2e7d32;background:#f1f8e9}}
  label.cand input{{margin:0}}
  label.cand img{{width:160px;height:160px;object-fit:cover;border-radius:6px;
                  box-shadow:0 1px 4px rgba(0,0,0,.15)}}
  span.idx{{font-size:10px;color:#777;font-weight:600}}
  .legend{{font-size:11px;color:#888;margin-bottom:14px}}
  .legend span{{display:inline-block;width:12px;height:12px;border-radius:2px;
                background:#e8f5e9;border:2px solid #2e7d32;vertical-align:middle;margin-right:4px}}
  .actions{{margin:24px 0}}
  .btn{{padding:12px 36px;border:none;border-radius:8px;font-size:15px;font-weight:700;
         cursor:pointer;background:#2e7d32;color:#fff;transition:opacity .2s}}
  .btn:hover{{opacity:.88}}
  .note{{font-size:11px;color:#999;margin-top:8px}}
</style>
</head>
<body>
<h1>Image Approval</h1>
<p class="sub">Review the VLM-selected candidate for each shot. Override by clicking a different image, then click Approve.</p>
<div class="legend"><span></span> VLM pick (green border). Click any image to select a different candidate.</div>

<form id="af">
<table>
<thead><tr><th>Shot</th><th>Candidates (select one per shot)</th></tr></thead>
<tbody>{rows}</tbody>
</table>

<div class="actions">
  <button type="button" class="btn" onclick="approve()">✓ Approve &amp; start video generation</button>
  <p class="note">Selections are saved to the critique feedback log so the AI learns your preferences.</p>
</div>
</form>

<script>
// Highlight selected candidate on click
document.querySelectorAll('label.cand').forEach(lbl => {{
  lbl.addEventListener('click', () => {{
    const name = lbl.querySelector('input').name;
    document.querySelectorAll(`input[name="${{name}}"]`).forEach(r => {{
      r.closest('label').classList.remove('winner');
    }});
    lbl.classList.add('winner');
  }});
}});

function approve() {{
  const picks = {{}};
  document.querySelectorAll('input[type=radio]:checked').forEach(r => {{
    const shotId = r.name.replace('pick_', '');
    picks[shotId] = parseInt(r.value);
  }});
  fetch('/approve', {{
    method: 'POST',
    headers: {{'Content-Type': 'application/json'}},
    body: JSON.stringify({{picks}}),
  }}).then(() => {{
    document.body.innerHTML = '<div style="text-align:center;margin-top:120px">'
      + '<h2 style="color:#2e7d32">✓ Approved — video generation will start shortly.</h2></div>';
  }});
}}
</script>
</body>
</html>"""


# ── HTTP handler ───────────────────────────────────────────────────────────────

class _Handler(BaseHTTPRequestHandler):
    html: str = ""
    flag_path: Path = Path("image_approval.flag")

    def log_message(self, fmt, *args):
        pass

    def do_GET(self):
        if self.path == "/":
            body = self.html.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path.startswith("/img/"):
            try:
                path_b64 = self.path[5:]
                path = base64.urlsafe_b64decode(path_b64.encode()).decode()
                data = Path(path).read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            except Exception:
                self.send_response(404)
                self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == "/approve":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            try:
                data = json.loads(body) if body else {}
            except Exception:
                data = {}
            picks = data.get("picks", {})
            self.flag_path.write_text("approved:" + json.dumps(picks))
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"ok":true}')
        else:
            self.send_response(404)
            self.end_headers()


# ── Adapter ────────────────────────────────────────────────────────────────────

class ImageApprovalAdapter:
    """
    Serves a grid web UI on localhost:<port> showing all shots' candidate images.
    The operator confirms or overrides the VLM pick per shot, then clicks Approve.

    Returns ImageApprovalResult with final approved URIs and any override map.
    Writes overrides back to the critique feedback log so the VLM learns.
    """

    def __init__(
        self,
        work_dir: Path,
        feedback_log_path: Path,
        port: int = 8766,
        timeout_hours: float = 24.0,
        auto_approve: bool = False,
    ) -> None:
        self._work_dir = work_dir
        self._feedback_log_path = feedback_log_path
        self._port = port
        self._timeout = timeout_hours * 3600
        self._auto_approve = auto_approve

    async def health(self):
        from core.models.common import HealthStatus
        return HealthStatus(status="ok")

    async def estimate_cost(self, req):
        from core.models.common import CostEstimate
        return CostEstimate(amount=0.0, notes="Local web UI — no cost.")

    async def run(self, req: ImageApprovalRequest) -> ImageApprovalResult:
        if self._auto_approve:
            logger.info("image approval gate: auto-approved (VIDEO_ME_AUTO_APPROVE_IMAGES=true)")
            return ImageApprovalResult(
                approved_uris=[r.winner_uri for r in req.critique_results],
            )

        flag_path = self._work_dir / _FLAG_FILE
        flag_path.unlink(missing_ok=True)

        html = _render_html(req)
        server = self._start_server(html, flag_path)

        logger.info(
            "Image candidates ready for review → http://localhost:%d  (waiting up to %.0fh)",
            self._port, self._timeout / 3600,
        )
        print(f"\n{'='*60}")
        print(f"  IMAGE APPROVAL REQUIRED")
        print(f"  Open: http://localhost:{self._port}")
        print(f"  Review all shot candidates and click Approve.")
        print(f"{'='*60}\n")

        try:
            flag_content = await self._poll_flag(flag_path)
        finally:
            server.shutdown()

        # Parse picks: {"s01": 2, "s02": 0, ...}
        picks: dict[str, int] = {}
        if flag_content.startswith("approved:"):
            try:
                picks = json.loads(flag_content[len("approved:"):])
            except Exception:
                pass

        # Build final approved URIs, detect overrides
        approved_uris = []
        overrides: dict[str, int] = {}

        # Build a map from shot_id to critique result and candidate uris
        # req.shots may be Shot objects
        shot_ids = [getattr(s, "shot_id", str(s)) for s in req.shots]

        for shot_id, critique in zip(shot_ids, req.critique_results):
            human_pick = picks.get(shot_id)
            if human_pick is None:
                human_pick = critique.winner_index

            # Resolve URI — need original candidate list
            candidate_uris = getattr(critique, "candidate_uris", [critique.winner_uri])
            human_pick = max(0, min(human_pick, len(candidate_uris) - 1))
            approved_uri = candidate_uris[human_pick]
            approved_uris.append(approved_uri)

            if human_pick != critique.winner_index:
                overrides[shot_id] = human_pick
                logger.info("image approval: override shot %s → candidate %d (was %d)",
                            shot_id, human_pick, critique.winner_index)

        if overrides:
            self._record_overrides(shot_ids, req.critique_results, picks)

        log_event(logger, "image_approval_completed",
                  total_shots=len(shot_ids), overrides=len(overrides))
        return ImageApprovalResult(approved_uris=approved_uris, overrides=overrides)

    # ── private ──

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
            f"No image approval received within {self._timeout / 3600:.0f}h — job timed out."
        )

    def _record_overrides(
        self,
        shot_ids: list[str],
        critique_results: list[ImageCritiqueResult],
        picks: dict[str, int],
    ) -> None:
        """Update the last feedback entry for each overridden shot with human_override."""
        if not self._feedback_log_path.exists():
            return
        lines = self._feedback_log_path.read_text(encoding="utf-8").splitlines()
        updated: list[str] = []
        # Track which shot_ids we still need to patch (patch last matching entry)
        to_patch = {
            sid: picks[sid]
            for sid, cr in zip(shot_ids, critique_results)
            if picks.get(sid, cr.winner_index) != cr.winner_index
        }
        patched: set[str] = set()
        for line in reversed(lines):
            line = line.strip()
            if not line:
                updated.insert(0, line)
                continue
            try:
                entry = json.loads(line)
            except Exception:
                updated.insert(0, line)
                continue
            sid = entry.get("shot_id", "")
            if sid in to_patch and sid not in patched:
                entry["human_override"] = to_patch[sid]
                entry["override_reason"] = "human selected via approval UI"
                patched.add(sid)
                updated.insert(0, json.dumps(entry))
            else:
                updated.insert(0, line)
        self._feedback_log_path.write_text(
            "\n".join(updated) + "\n", encoding="utf-8"
        )
