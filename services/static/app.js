/* video_me dashboard — vanilla JS, no build step */

const STALL_THRESHOLD_SEC = 120;
const POLL_INTERVAL_MS    = 3000;
const DASHBOARD_TIME_ZONE = "America/Los_Angeles";

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

function formatTime(isoStr) {
  if (!isoStr) return "—";
  const d = new Date(isoStr);
  if (isNaN(d)) return isoStr;
  return new Intl.DateTimeFormat("en-GB", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
    timeZone: DASHBOARD_TIME_ZONE,
  }).format(d);
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
let _seenEventIds = new Set();
let _currentJobId = null;
let _wasTerminal = false;  // true if job was already terminal when page loaded

const TERMINAL_STATUSES = new Set(["completed", "failed", "blocked", "cancelled"]);

function initJobDetail(jobId, initialStatus) {
  _currentJobId = jobId;
  seedEventCursorFromDom();
  _wasTerminal = TERMINAL_STATUSES.has(initialStatus);
  if (_wasTerminal) return;  // already done — no SSE, no auto-reload loop
  connectSSE(jobId);
}

function seedEventCursorFromDom() {
  _seenEventIds = new Set();
  _lastEventId = 0;
  document.querySelectorAll("#events-feed [data-event-id]").forEach((row) => {
    const id = Number(row.dataset.eventId || 0);
    if (!Number.isFinite(id) || id <= 0) return;
    _seenEventIds.add(id);
    _lastEventId = Math.max(_lastEventId, id);
  });
}

function connectSSE(jobId) {
  if (_sse) { _sse.close(); _sse = null; }

  const url = `/api/jobs/${jobId}/stream?after_event_id=${_lastEventId}`;
  _sse = new EventSource(url);

  _sse.onmessage = (e) => {
    try {
      const ev = JSON.parse(e.data);
      const appended = appendEvent(ev);
      const eventId = Number(ev.event_id || 0);
      if (Number.isFinite(eventId) && eventId > 0) {
        _lastEventId = Math.max(_lastEventId, eventId);
      }
      // Live-update the stage timeline dots.
      if (appended && ev.stage_name && ev.event_type) {
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
  const eventId = Number(ev.event_id || 0);
  if (Number.isFinite(eventId) && eventId > 0 && _seenEventIds.has(eventId)) {
    return false;
  }
  if (!feed) return false;

  const row = document.createElement("div");
  row.className = "ev-row";
  if (Number.isFinite(eventId) && eventId > 0) {
    row.dataset.eventId = String(eventId);
    _seenEventIds.add(eventId);
  }

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
  return true;
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
// Jobs-list bulk actions (select, cancel, delete)
// ---------------------------------------------------------------------------

function _selectedJobIds() {
  return Array.from(document.querySelectorAll(".job-select-checkbox:checked")).map((cb) => cb.value);
}

function toggleSelectAllJobs(headerCheckbox) {
  document.querySelectorAll(".job-select-checkbox").forEach((cb) => {
    cb.checked = headerCheckbox.checked;
  });
  onJobSelectionChange();
}

function clearJobSelection() {
  document.querySelectorAll(".job-select-checkbox").forEach((cb) => { cb.checked = false; });
  const selectAll = document.getElementById("jobs-select-all");
  if (selectAll) selectAll.checked = false;
  onJobSelectionChange();
}

function onJobSelectionChange() {
  const ids = _selectedJobIds();
  const toolbar = document.getElementById("jobs-bulk-toolbar");
  const count = document.getElementById("jobs-bulk-count");
  if (toolbar) toolbar.style.display = ids.length ? "flex" : "none";
  if (count) count.textContent = `${ids.length} selected`;
}

async function _postBulkAction(url, jobIds) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ job_ids: jobIds }),
  });
  if (!res.ok) {
    const data = await res.json().catch(() => ({}));
    throw new Error(data.detail?.message || `${url} failed (${res.status})`);
  }
  return (await res.json()).results || {};
}

function _summarizeBulkResults(results, resultLabels) {
  const counts = {};
  Object.values(results).forEach((r) => { counts[r] = (counts[r] || 0) + 1; });
  return Object.entries(counts)
    .map(([r, n]) => `${n} ${resultLabels[r] || r}`)
    .join(", ");
}

async function bulkCancelSelected() {
  const ids = _selectedJobIds();
  if (!ids.length) return;
  if (!confirm(`Cancel ${ids.length} selected job(s)?\n\nEach worker stops after its current stage finishes.`)) return;
  try {
    const results = await _postBulkAction("/api/jobs/bulk-cancel", ids);
    alert(_summarizeBulkResults(results, {
      cancel_requested: "cancellation requested",
      already_terminal: "already finished (skipped)",
      not_found: "not found",
    }));
    location.reload();
  } catch (err) {
    alert("Error: " + err.message);
  }
}

async function bulkDeleteSelected() {
  const ids = _selectedJobIds();
  if (!ids.length) return;
  if (!confirm(
    `Delete ${ids.length} selected job(s) from the list?\n\n` +
    "Only finished jobs (completed/failed/cancelled/blocked) are removed — " +
    "active jobs are skipped. This only removes the dashboard record, not " +
    "rendered files on disk."
  )) return;
  try {
    const results = await _postBulkAction("/api/jobs/bulk-delete", ids);
    alert(_summarizeBulkResults(results, {
      deleted: "deleted",
      skipped_active: "still active (cancel first)",
      not_found: "not found",
    }));
    location.reload();
  } catch (err) {
    alert("Error: " + err.message);
  }
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

// ---------------------------------------------------------------------------
// Chat drawer — Pipeline Assistant
// ---------------------------------------------------------------------------

let _chatJobId = null;
let _chatStreaming = false;

function _getChatJobId() {
  const el = document.querySelector("[data-job-id]");
  return el ? el.dataset.jobId : null;
}

function openChatDrawer() {
  const drawer = document.getElementById("chat-drawer");
  if (!drawer) return;
  drawer.style.display = "flex";

  // Push main content right so job pane stays fully visible
  const main = document.querySelector(".main");
  if (main) main.style.marginLeft = "360px";

  _chatJobId = _getChatJobId();
  const label = document.getElementById("chat-job-label");
  if (label) {
    if (_chatJobId) {
      const short = _chatJobId.length > 26 ? _chatJobId.slice(0, 26) + "…" : _chatJobId;
      label.textContent = short;
      label.title = _chatJobId;
    } else {
      label.textContent = "no job selected";
      label.title = "";
    }
  }

  if (_chatJobId) {
    loadChatHistory(_chatJobId);
  } else {
    const msgs = document.getElementById("chat-messages");
    if (msgs) msgs.innerHTML = '<div class="chat-empty">Open a job to start chatting about it.</div>';
  }
  setTimeout(() => {
    const input = document.getElementById("chat-input");
    if (input) input.focus();
  }, 80);
}

function closeChatDrawer() {
  const drawer = document.getElementById("chat-drawer");
  if (drawer) drawer.style.display = "none";
  // Restore main content width
  const main = document.querySelector(".main");
  if (main) main.style.marginLeft = "";
}

async function loadChatHistory(jobId) {
  const msgs = document.getElementById("chat-messages");
  if (!msgs) return;
  try {
    const res = await fetch(`/api/jobs/${jobId}/chat/history`);
    if (!res.ok) return;
    const data = await res.json();
    const history = data.messages || [];
    msgs.innerHTML = "";
    if (history.length === 0) {
      msgs.innerHTML = '<div class="chat-empty">Ask me anything about this job…</div>';
      return;
    }
    history.forEach((m) => _appendChatBubble(m.role, m.content, false));
  } catch (_) {}
}

function _appendChatBubble(role, content, streaming) {
  const msgs = document.getElementById("chat-messages");
  if (!msgs) return null;
  const empty = msgs.querySelector(".chat-empty");
  if (empty) empty.remove();

  const bubble = document.createElement("div");
  bubble.className = `chat-msg chat-msg--${role}`;
  if (streaming) bubble.dataset.streaming = "true";
  bubble.textContent = content || "";
  msgs.appendChild(bubble);
  msgs.scrollTop = msgs.scrollHeight;
  return bubble;
}

function _appendToken(token) {
  const msgs = document.getElementById("chat-messages");
  if (!msgs) return;
  const streaming = msgs.querySelector(".chat-msg[data-streaming='true']");
  if (!streaming) return;
  streaming.textContent += token;
  msgs.scrollTop = msgs.scrollHeight;
}

function _appendToolBadge(name, result) {
  const msgs = document.getElementById("chat-messages");
  if (!msgs) return;
  const badge = document.createElement("div");
  badge.className = "chat-tool-badge";
  badge.textContent = `⚙ ${name}`;
  badge.title = JSON.stringify(result, null, 2);
  msgs.appendChild(badge);
  msgs.scrollTop = msgs.scrollHeight;
}

function _finalizeStreaming() {
  const msgs = document.getElementById("chat-messages");
  if (!msgs) return;
  const streaming = msgs.querySelector(".chat-msg[data-streaming='true']");
  if (streaming) streaming.removeAttribute("data-streaming");
}

async function sendChatMessage() {
  if (!_chatJobId || _chatStreaming) return;
  const input = document.getElementById("chat-input");
  const sendBtn = document.getElementById("chat-send-btn");
  if (!input || !sendBtn) return;

  const message = input.value.trim();
  if (!message) return;

  input.value = "";
  _chatStreaming = true;
  sendBtn.disabled = true;

  _appendChatBubble("user", message, false);
  const assistantBubble = _appendChatBubble("assistant", "", true);

  try {
    const res = await fetch(`/api/jobs/${_chatJobId}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message }),
    });

    if (!res.ok || !res.body) {
      if (assistantBubble) assistantBubble.textContent = "Error: could not reach assistant.";
      _finalizeStreaming();
      return;
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      // SSE chunks are separated by \n\n
      const chunks = buffer.split("\n\n");
      buffer = chunks.pop() || "";

      for (const chunk of chunks) {
        if (!chunk.startsWith("data: ")) continue;
        try {
          const payload = JSON.parse(chunk.slice(6));
          if (payload.type === "token") {
            _appendToken(payload.content);
          } else if (payload.type === "tool_call") {
            _appendToolBadge(payload.name, payload.result);
          } else if (payload.type === "done") {
            _finalizeStreaming();
          } else if (payload.type === "error") {
            if (assistantBubble) assistantBubble.textContent = `Error: ${payload.message}`;
            _finalizeStreaming();
          }
        } catch (_) {}
      }
    }
  } catch (err) {
    if (assistantBubble) assistantBubble.textContent = `Error: ${err.message}`;
  } finally {
    _chatStreaming = false;
    sendBtn.disabled = false;
    _finalizeStreaming();
    input.focus();
  }
}

async function clearChatHistory() {
  if (!_chatJobId) return;
  try {
    await fetch(`/api/jobs/${_chatJobId}/chat/history`, { method: "DELETE" });
    const msgs = document.getElementById("chat-messages");
    if (msgs) msgs.innerHTML = '<div class="chat-empty">Ask me anything about this job…</div>';
  } catch (_) {}
}

document.addEventListener("DOMContentLoaded", () => {
  document.getElementById("chat-toggle-btn")?.addEventListener("click", openChatDrawer);
  document.getElementById("chat-close-btn")?.addEventListener("click", closeChatDrawer);
  document.getElementById("chat-overlay")?.addEventListener("click", closeChatDrawer);
  document.getElementById("chat-send-btn")?.addEventListener("click", sendChatMessage);

  const chatInput = document.getElementById("chat-input");
  if (chatInput) {
    chatInput.addEventListener("keydown", (e) => {
      if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
        e.preventDefault();
        sendChatMessage();
      }
    });
  }
});
