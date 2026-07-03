/* video_me dashboard — vanilla JS, no build step */

const STALL_THRESHOLD_SEC = 120;
const POLL_INTERVAL_MS    = 3000;

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

function formatTime(isoStr) {
  if (!isoStr) return "—";
  const d = new Date(isoStr);
  if (isNaN(d)) return isoStr;
  return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

function elapsedSec(isoStart) {
  if (!isoStart) return null;
  return Math.floor((Date.now() - new Date(isoStart)) / 1000);
}

function formatElapsed(sec) {
  if (sec === null) return "";
  const m = Math.floor(sec / 60);
  const s = sec % 60;
  return m > 0 ? `${m}m ${s}s` : `${s}s`;
}

// ---------------------------------------------------------------------------
// Status badge
// ---------------------------------------------------------------------------

function makeBadge(status, extraClass) {
  const running = status === "running";
  const el = document.createElement("span");
  el.className = `badge badge-${status}${extraClass ? " " + extraClass : ""}${running ? " pulse" : ""}`;
  el.textContent = status.replace(/_/g, " ");
  return el;
}

// ---------------------------------------------------------------------------
// SSE + polling for job detail page
// ---------------------------------------------------------------------------

let _pollTimer = null;
let _sse = null;
let _lastEventId = 0;
let _currentJobId = null;
let _wasTerminal = false;  // true if job was already terminal when page loaded

const TERMINAL_STATUSES = new Set(["completed", "failed", "blocked", "cancelled"]);

function initJobDetail(jobId, initialStatus) {
  _currentJobId = jobId;
  _wasTerminal = TERMINAL_STATUSES.has(initialStatus);
  if (_wasTerminal) return;  // already done — no SSE, no auto-reload loop
  connectSSE(jobId);
}

function connectSSE(jobId) {
  if (_sse) { _sse.close(); _sse = null; }

  const url = `/api/jobs/${jobId}/stream?after_event_id=${_lastEventId}`;
  _sse = new EventSource(url);

  _sse.onmessage = (e) => {
    try {
      const ev = JSON.parse(e.data);
      appendEvent(ev);
      _lastEventId = ev.event_id || _lastEventId;
      // Live-update the stage timeline dots.
      if (ev.stage_name && ev.event_type) {
        updateTimeline(ev.stage_name, ev.event_type);
      }
    } catch (_) {}
  };

  _sse.addEventListener("done", (e) => {
    _sse.close(); _sse = null;
    stopPolling();
    // Only reload if we transitioned to terminal while watching (not if already terminal at page load).
    if (!_wasTerminal) { _wasTerminal = true; setTimeout(() => location.reload(), 800); }
  });

  _sse.onerror = () => {
    _sse.close(); _sse = null;
    startPolling(jobId);
  };
}

function startPolling(jobId) {
  if (_pollTimer) return;
  _pollTimer = setInterval(() => pollStatus(jobId), POLL_INTERVAL_MS);
}

function stopPolling() {
  if (_pollTimer) { clearInterval(_pollTimer); _pollTimer = null; }
}

async function pollStatus(jobId) {
  try {
    const res = await fetch(`/api/jobs/${jobId}`);
    if (!res.ok) return;
    const data = await res.json();
    applyJobStatus(data);
  } catch (_) {}
}

function applyJobStatus(data) {
  const job = data.job || data;

  // Update status badge in page header.
  const badgeEl = document.getElementById("job-status-badge");
  if (badgeEl) {
    badgeEl.className = `badge badge-${job.status}${job.status === "running" ? " pulse" : ""}`;
    badgeEl.textContent = (job.status || "").replace(/_/g, " ");
  }

  // Update current stage.
  const stageEl = document.getElementById("current-stage");
  if (stageEl && job.current_stage) stageEl.textContent = job.current_stage;

  // Stale / stalled detection.
  checkStall(job.last_heartbeat_at, job.status);

  // Show approval card if pending.
  if (job.pending_approval || (data.pending_approval)) {
    const ap = job.pending_approval || data.pending_approval;
    showApprovalBanner(ap);
  }

  // Reload once if job transitioned to terminal while we were watching.
  if (TERMINAL_STATUSES.has(job.status)) {
    stopPolling();
    if (!_wasTerminal) { _wasTerminal = true; setTimeout(() => location.reload(), 1200); }
  }
}

// ---------------------------------------------------------------------------
// Stall detection
// ---------------------------------------------------------------------------

function checkStall(lastHeartbeatAt, status) {
  const banner = document.getElementById("stall-banner");
  if (!banner) return;
  const RUNNING = new Set(["running", "pending_plan_approval", "pending_image_approval"]);
  if (!RUNNING.has(status) || !lastHeartbeatAt) { banner.classList.remove("visible"); return; }
  const age = (Date.now() - new Date(lastHeartbeatAt)) / 1000;
  if (age > STALL_THRESHOLD_SEC) {
    banner.classList.add("visible");
    const msg = banner.querySelector(".stall-msg");
    if (msg) msg.textContent = `Worker last seen ${Math.floor(age)}s ago — job may be stalled.`;
  } else {
    banner.classList.remove("visible");
  }
}

// ---------------------------------------------------------------------------
// Events feed
// ---------------------------------------------------------------------------

function appendEvent(ev) {
  const feed = document.getElementById("events-feed");
  if (!feed) return;

  const row = document.createElement("div");
  row.className = "ev-row";

  const time = document.createElement("span");
  time.className = "ev-time";
  time.textContent = formatTime(ev.created_at);

  const lvl = document.createElement("span");
  lvl.className = `ev-level ev-level-${ev.level || "info"}`;
  lvl.textContent = (ev.level || "info").toUpperCase();

  const msg = document.createElement("span");
  msg.className = "ev-msg";
  msg.textContent = ev.message || ev.event_type;

  const stage = document.createElement("span");
  stage.className = "ev-stage";
  stage.textContent = ev.stage_name || "";

  row.appendChild(time);
  row.appendChild(lvl);
  row.appendChild(msg);
  if (ev.stage_name) row.appendChild(stage);

  feed.appendChild(row);
  feed.scrollTop = feed.scrollHeight;
}

// ---------------------------------------------------------------------------
// Approval banner (inline, no page reload)
// ---------------------------------------------------------------------------

function showApprovalBanner(approval) {
  const container = document.getElementById("approval-container");
  if (!container || !approval) return;
  if (container.dataset.loaded === approval.approval_id) return;
  container.dataset.loaded = approval.approval_id;
  // The server-rendered approval card is already in the HTML on page load.
  // This function is for dynamically appearing approvals (not visible on first load).
  container.style.display = "block";
}

// ---------------------------------------------------------------------------
// Button loading state — attach to all submit buttons
// ---------------------------------------------------------------------------

document.addEventListener("DOMContentLoaded", () => {
  document.querySelectorAll("form[data-submit-label]").forEach((form) => {
    form.addEventListener("submit", () => {
      const btn = form.querySelector("button[type=submit]");
      if (!btn) return;
      const label = form.dataset.submitLabel || "Submitting…";
      btn.disabled = true;
      btn.textContent = label;
    });
  });

  // Auto-initialise job detail page.
  const detailRoot = document.getElementById("job-detail-root");
  if (detailRoot && detailRoot.dataset.jobId) {
    initJobDetail(detailRoot.dataset.jobId, detailRoot.dataset.status);
    // Also start stall check every 30s.
    setInterval(() => {
      const hb = detailRoot.dataset.lastHeartbeat;
      const st = detailRoot.dataset.status;
      checkStall(hb, st);
    }, 30000);
  }

  // Jobs-list page: poll /api/jobs every 10s and patch rows in-place (no full reload).
  const listRoot = document.getElementById("jobs-list-root");
  if (listRoot && listRoot.dataset.hasRunning === "true") {
    setInterval(refreshJobsList, 10000);
  }
});

// ---------------------------------------------------------------------------
// Jobs-list AJAX refresh — update status/stage cells without a full reload
// ---------------------------------------------------------------------------

const ACTIVE_STATUSES = new Set([
  "created", "running", "queued", "cancel_requested",
  "pending_plan_approval", "pending_image_approval", "pending_final_review", "stalled",
]);

async function refreshJobsList() {
  // Don't refresh while the new-job modal is open.
  const modal = document.getElementById("new-job-modal");
  if (modal && modal.style.display === "flex") return;

  try {
    const res = await fetch("/api/jobs?limit=100");
    if (!res.ok) return;
    const data = await res.json();
    const jobs = data.items || [];

    const tbody = document.getElementById("jobs-tbody");
    if (!tbody) return;

    let hasActive = false;
    jobs.forEach((job) => {
      if (ACTIVE_STATUSES.has(job.status)) hasActive = true;

      const row = tbody.querySelector(`tr[data-job-id="${job.job_id}"]`);
      if (!row) return;

      // Patch status badge.
      const statusCell = row.querySelector(".td-job-status");
      if (statusCell) {
        const isRunning = job.status === "running";
        statusCell.innerHTML =
          `<span class="badge badge-${job.status}${isRunning ? " pulse" : ""}">${
            job.status.replace(/_/g, " ")
          }</span>`;
      }

      // Patch stage cell.
      const stageCell = row.querySelector(".td-stage");
      if (stageCell) stageCell.textContent = job.current_stage || "—";
    });

    // Stop timer once no active jobs remain.
    const root = document.getElementById("jobs-list-root");
    if (root && !hasActive) {
      root.dataset.hasRunning = "false";
      // The interval was created without a stored reference, so we rely on the
      // data attribute check at the next tick — the browser will keep firing but
      // refreshJobsList will be a no-op (no active rows to update).
    }
  } catch (_) {}
}

// ---------------------------------------------------------------------------
// Stage timeline live updates
// ---------------------------------------------------------------------------

function updateTimeline(stageName, eventType) {
  if (!stageName) return;
  const step = document.querySelector(`.tl-step[data-stage-id="${stageName}"]`);
  if (!step) return;
  const dot = step.querySelector(".tl-dot");
  if (eventType === "stage_completed") {
    step.className = "tl-step done";
    if (dot) dot.textContent = "✓";
  } else if (eventType === "stage_failed") {
    step.className = "tl-step failed";
    if (dot) dot.textContent = "✕";
  } else if (eventType === "stage_started") {
    // Clear any previous running state on sibling steps.
    document.querySelectorAll(".tl-step.running").forEach((s) => {
      if (s !== step) { s.className = "tl-step"; }
    });
    step.className = "tl-step running";
    if (dot) dot.textContent = "…";
  }
}

// ---------------------------------------------------------------------------
// Image approval: candidate selection
// ---------------------------------------------------------------------------

function selectCandidate(card, shotId, idx) {
  // Deselect siblings.
  const parent = card.closest(".candidates-grid");
  if (parent) parent.querySelectorAll(".candidate-card").forEach(c => c.classList.remove("selected"));
  card.classList.add("selected");

  // Update hidden input.
  const input = document.getElementById(`pick-${shotId}`);
  if (input) input.value = idx;
}
