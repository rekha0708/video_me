/* Pipeline Studio — dependency-free controller for pipeline jobs. */
(function () {
  "use strict";

  const root = document.getElementById("pipeline-studio");
  if (!root) return;

  /* ── helpers ────────────────────────────────────────────────────────── */

  const $ = (sel, scope) => (scope || document).querySelector(sel);
  const $$ = (sel, scope) => Array.from((scope || document).querySelectorAll(sel));
  const selectedValue = (name) => {
    const el = $(`input[name="${name}"]:checked`);
    return el ? el.value : "";
  };

  function escapeHtml(v) {
    return String(v)
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;");
  }

  async function requestJson(url, opts) {
    const res = await fetch(url, opts);
    const data = await res.json();
    if (!res.ok) {
      let msg;
      if (Array.isArray(data.detail)) {
        msg = data.detail.map((e) => {
          const field = e.loc ? e.loc.slice(1).join(".") : "";
          return field ? `${field}: ${e.msg}` : e.msg;
        }).join("\n");
      } else {
        msg = data.detail?.message || JSON.stringify(data.detail) || "Unknown error";
      }
      throw new Error(msg);
    }
    return data;
  }

  /* ── constants ──────────────────────────────────────────────────────── */

  const URLS = {
    options: root.dataset.optionsUrl || "/api/pipeline/options",
    create: root.dataset.createUrl || "/api/jobs",
    localVideos: "/api/local-videos",
    uploadCharImage: "/api/uploads/character-image",
    uploadLoraImage: "/api/uploads/lora-training-image",
  };

  const PHASE_HINTS = {
    noop: "Runs a mock pipeline in-memory — no services needed.",
    transcribe: "Downloads the video, transcribes audio, and analyzes content.",
    script_plan: "Adapts the transcript into a script, plans shots. Needs completed transcribe.",
    render: "Renders character images, synthesizes voice, generates video per shot.",
    assemble: "Concatenates rendered shots into a final video and publishes.",
    all: "Full end-to-end pipeline — all stages in sequence.",
    lora_train: "Uploads training image(s), then launches Flux LoRA training.",
  };

  const INPUT_LABELS = {
    url: "Video URL",
    file: "Local File",
    story: "Story",
    story_images: "Story + Images",
    lora_training: "LoRA Training",
  };

  const VIDEO_LABELS = {
    "": "Default",
    wan_s2v: "Wan 2.2 S2V",
    wan: "Wan 2.2 I2V",
    wan_lightx2v: "LightX2V",
    wan_animate: "Wan Animate",
    ltx: "LTX-Video",
  };

  const LIPSYNC_LABELS = {
    "": "Default",
    latentsync: "LatentSync",
    musetalk: "MuseTalk",
    none: "Skip",
  };

  const RENDER_LABELS = { full: "Full", source_audio: "Source Audio", re_voice: "Re-voice" };
  const LANG_LABELS = { en: "English", hi: "Hindi", both: "Both (EN + HI)" };
  const STORY_MODES = new Set(["story", "story_images"]);
  const RESTRICTED_PHASES = new Set(["script_plan", "render", "assemble"]);
  const GPU_PHASES = new Set(["render", "all"]);

  /* ── mutable state ──────────────────────────────────────────────────── */

  const state = {
    options: null,
    casts: [],
    currentCastMembers: [],
    serviceReadiness: null,
    defaults: null,
    submitting: false,
  };

  /* ── options / casts ────────────────────────────────────────────────── */

  async function loadOptions() {
    try {
      const data = await requestJson(URLS.options);
      state.options = data;
      state.casts = data.casts || [];
      state.serviceReadiness = data.readiness || null;
      state.defaults = data.defaults || {};
      populateCasts(data.default || root.dataset.defaultCast);
    } catch (e) {
      console.error("Pipeline options fetch failed:", e);
      try {
        const fallback = await requestJson("/api/casts");
        state.casts = fallback.casts || [];
        populateCasts(fallback.default || root.dataset.defaultCast);
      } catch (_) { /* degrade gracefully */ }
    }
    renderAll();
  }

  function populateCasts(defaultId) {
    const sel = $("#pipeline-cast");
    sel.innerHTML = state.casts.map((c) =>
      `<option value="${escapeHtml(c.id)}"${c.id === defaultId ? " selected" : ""}>`
      + `${escapeHtml(c.id)} — ${c.members.map((m) => m.name).join(", ")} (${c.member_count})`
      + `</option>`
    ).join("");
    sel.disabled = false;
    onCastChange(sel.value);
  }

  function onCastChange(castId) {
    const cast = state.casts.find((c) => c.id === castId);
    state.currentCastMembers = cast ? cast.members : [];
    rebuildImageSlots(state.currentCastMembers);
    rebuildLoraMemberOptions(state.currentCastMembers);
    updateStoryPlaceholder(state.currentCastMembers);
    updateCastCapabilities(state.currentCastMembers);
    renderAll();
  }

  function updateCastCapabilities(members) {
    const el = $("#pipeline-cast-capabilities");
    if (!el || !members.length) { if (el) el.textContent = ""; return; }
    el.innerHTML = members.map((m) => {
      const lora = m.has_lora ? "&#10003; LoRA" : "&#10007; LoRA";
      const voice = m.has_voice ? "&#10003; Voice" : "&#10007; Voice";
      return `<span style="margin-right:12px"><strong>${escapeHtml(m.name)}</strong>: ${lora}, ${voice}</span>`;
    }).join("");
  }

  /* ── source mode ────────────────────────────────────────────────────── */

  function currentSourceKind() {
    return selectedValue("source_kind");
  }

  function toggleSourceMode(mode) {
    ["url", "file", "story", "story_images", "lora_training"].forEach((m) => {
      const panel = $(`#panel-${m}`);
      if (panel) panel.hidden = m !== mode;
    });

    const isStory = STORY_MODES.has(mode);
    const isLora = mode === "lora_training";

    // Phase restrictions
    RESTRICTED_PHASES.forEach((p) => {
      const opt = $(`#opt-${p}`);
      if (opt) opt.disabled = isStory || isLora;
    });
    ["noop", "transcribe", "all"].forEach((p) => {
      const opt = $(`#pipeline-phase option[value="${p}"]`);
      if (opt) opt.disabled = isLora && p !== "noop";
    });
    const loraOpt = $("#opt-lora_train");
    if (loraOpt) loraOpt.disabled = !isLora;

    const transcribeOpt = $("#opt-transcribe");
    if (transcribeOpt) {
      transcribeOpt.textContent = isStory
        ? "transcribe — analyze story (review segmentation)"
        : "transcribe — fetch → transcribe → analyze content";
    }

    const sel = $("#pipeline-phase");
    if (isLora) sel.value = "lora_train";
    if (isStory && RESTRICTED_PHASES.has(sel.value)) sel.value = "all";
    if (!isLora && sel.value === "lora_train") sel.value = "all";

    // Render mode restrictions
    const noSourceAudio = isStory || isLora;
    $$('#pipeline-render-heading').forEach(() => {}); // noop, just for readability
    $$('input[name="render_mode"]').forEach((inp) => {
      const label = inp.closest("label");
      if (noSourceAudio && inp.value !== "full") {
        inp.disabled = true;
        if (label) label.style.opacity = "0.4";
      } else {
        inp.disabled = false;
        if (label) label.style.opacity = "";
      }
    });
    if (noSourceAudio) {
      const fullRadio = $('input[name="render_mode"][value="full"]');
      if (fullRadio) fullRadio.checked = true;
    }

    // Audio profile
    const audioProfile = $("#pipeline-audio-profile");
    if (audioProfile) {
      audioProfile.disabled = noSourceAudio;
      if (noSourceAudio) audioProfile.value = "auto";
    }
    updateIsolateVocals();

    // Source language
    const srcLang = $("#pipeline-whisper-lang");
    if (srcLang) {
      srcLang.disabled = noSourceAudio;
      if (noSourceAudio) srcLang.value = "auto";
    }

    // Rights label
    const rightsLabel = $("#pipeline-rights-label");
    if (rightsLabel) {
      rightsLabel.textContent = isLora
        ? "I confirm I own/have rights to these training images."
        : isStory
        ? "I confirm I own/have rights to this story and images."
        : "I confirm this source is cleared for transformative use.";
    }

    if (mode === "file") loadLocalVideos();
    renderAll();
  }

  function updateIsolateVocals() {
    const ap = $("#pipeline-audio-profile");
    const cb = $("#pipeline-isolate-vocals");
    if (!ap || !cb) return;
    const ok = !ap.disabled && ap.value === "singing";
    cb.disabled = !ok;
    if (!ok) cb.checked = false;
  }

  /* ── local files ────────────────────────────────────────────────────── */

  async function loadLocalVideos() {
    const sel = $("#pipeline-file");
    const hint = $("#pipeline-file-dir-hint");
    const dirInput = $("#pipeline-file-dir");
    const dir = dirInput.value.trim();
    const url = dir ? `/api/local-videos?dir=${encodeURIComponent(dir)}` : "/api/local-videos";
    sel.innerHTML = '<option value="">Loading…</option>';
    try {
      const data = await requestJson(url);
      if (!dir) dirInput.value = data.dir;
      if (data.error) {
        hint.textContent = data.error;
        hint.style.color = "var(--color-error, #e53e3e)";
        sel.innerHTML = '<option value="">—</option>';
        return;
      }
      hint.textContent = data.videos.length + " video(s) found";
      hint.style.color = "";
      if (!data.videos.length) {
        sel.innerHTML = '<option value="">No video files found</option>';
        return;
      }
      sel.innerHTML = '<option value="">— select a file —</option>' +
        data.videos.map((v) =>
          `<option value="${escapeHtml(v.uri)}">${escapeHtml(v.name)} (${v.size_mb} MB)</option>`
        ).join("");
    } catch (e) {
      sel.innerHTML = '<option value="">Error loading files</option>';
      hint.textContent = e.message;
      hint.style.color = "var(--color-error, #e53e3e)";
    }
    renderAll();
  }

  /* ── image slots (story+images) ─────────────────────────────────────── */

  function rebuildImageSlots(members) {
    const container = $("#pipeline-images-slots");
    if (!container) return;
    container.innerHTML = members.map((m) =>
      `<div style="display:flex;align-items:center;gap:10px;margin-bottom:8px">`
      + `<span style="min-width:60px;font-weight:600">${escapeHtml(m.name)}</span>`
      + `<input type="file" id="pipeline-img-${escapeHtml(m.id)}" accept="image/png,image/jpeg,image/webp" data-member-id="${escapeHtml(m.id)}">`
      + `<span class="form-hint" id="pipeline-img-status-${escapeHtml(m.id)}"></span>`
      + `</div>`
    ).join("");
  }

  /* ── LoRA training ──────────────────────────────────────────────────── */

  function rebuildLoraMemberOptions(members) {
    const sel = $("#pipeline-lora-member");
    if (!sel) return;
    sel.innerHTML = members.map((m) =>
      `<option value="${escapeHtml(m.id)}">${escapeHtml(m.name)} (${escapeHtml(m.id)})</option>`
    ).join("");
    rebuildLoraCaptionRows();
  }

  function loraCaptionPlaceholder() {
    const memberId = $("#pipeline-lora-member")?.value || "";
    const castId = $("#pipeline-cast")?.value || "";
    const member = state.currentCastMembers.find((m) => m.id === memberId);
    const name = member ? member.name : memberId || "character";
    const trigger = memberId ? `${castId}_${memberId}`.toLowerCase() : "trigger_token";
    return `${trigger}, ${name}, describe outfit, pose, camera angle, background, lighting`;
  }

  function rebuildLoraCaptionRows() {
    const filesInput = $("#pipeline-lora-files");
    const container = $("#pipeline-lora-caption-rows");
    if (!filesInput || !container) return;
    const files = Array.from(filesInput.files || []);
    if (!files.length) {
      container.innerHTML = '<div class="form-hint">Choose image files to add captions.</div>';
      renderAll();
      return;
    }
    const placeholder = escapeHtml(loraCaptionPlaceholder());
    container.innerHTML = files.map((file, idx) =>
      `<div style="border:1px solid var(--border);border-radius:6px;padding:8px;margin-bottom:8px">`
      + `<div class="td-mono" style="font-size:12px;margin-bottom:6px">${escapeHtml(file.name)}</div>`
      + `<textarea id="pipeline-lora-caption-${idx}" rows="3" style="width:100%;font-family:monospace;font-size:12px"`
      + ` placeholder="${placeholder}"></textarea>`
      + `</div>`
    ).join("");
    renderAll();
  }

  function updateStoryPlaceholder(members) {
    ["pipeline-story-text", "pipeline-story-images-text"].forEach((id) => {
      const ta = $(`#${id}`);
      if (!ta || !members.length) return;
      const names = members.length > 1
        ? members[0].name + " and " + members[1].name
        : members[0].name;
      ta.placeholder = "Paste your story here.\n\nStructured format (optional):\n"
        + "0-4: " + names + " find a rainbow.\n"
        + "4-8: They count the colors together.\n\n"
        + "Or just paste free text — it will be auto-segmented.";
    });
  }

  /* ── uploads ────────────────────────────────────────────────────────── */

  async function uploadCharacterImage(memberId) {
    const input = $(`#pipeline-img-${memberId}`);
    if (!input || !input.files.length) return null;
    const formData = new FormData();
    formData.append("member_id", memberId);
    formData.append("file", input.files[0]);
    const castVal = $("#pipeline-cast")?.value;
    if (castVal) formData.append("cast_ref", castVal);
    const resp = await fetch(URLS.uploadCharImage, { method: "POST", body: formData });
    if (!resp.ok) {
      const data = await resp.json();
      throw new Error(data.detail?.message || "Upload failed for " + memberId);
    }
    const data = await resp.json();
    const statusEl = $(`#pipeline-img-status-${memberId}`);
    if (statusEl) statusEl.textContent = "✓";
    return { member_id: memberId, path: data.path };
  }

  async function uploadLoraTrainingImage(memberId, file, caption) {
    const formData = new FormData();
    formData.append("member_id", memberId);
    formData.append("file", file);
    if (caption) formData.append("caption", caption);
    const castVal = $("#pipeline-cast")?.value;
    if (castVal) formData.append("cast_ref", castVal);
    const resp = await fetch(URLS.uploadLoraImage, { method: "POST", body: formData });
    if (!resp.ok) {
      const data = await resp.json();
      throw new Error(data.detail?.message || "LoRA image upload failed");
    }
    const data = await resp.json();
    return data.path;
  }

  /* ── render: summary sidebar ────────────────────────────────────────── */

  function renderSummary() {
    const castSel = $("#pipeline-cast");
    const castId = castSel ? castSel.value : "";
    const cast = state.casts.find((c) => c.id === castId);
    $("#summary-cast").textContent = cast ? `${cast.id} (${cast.member_count})` : "Not selected";

    const mode = currentSourceKind();
    $("#summary-input").textContent = INPUT_LABELS[mode] || mode;

    // Source value
    let sourceText = "Not provided";
    if (mode === "url") {
      const v = ($("#pipeline-url")?.value || "").trim();
      sourceText = v ? (v.length > 40 ? v.slice(0, 37) + "…" : v) : "Not provided";
    } else if (mode === "file") {
      const v = $("#pipeline-file")?.value || "";
      const parts = v.split("/");
      sourceText = v ? parts[parts.length - 1] : "Not selected";
    } else if (mode === "story" || mode === "story_images") {
      const id = mode === "story" ? "pipeline-story-text" : "pipeline-story-images-text";
      const v = ($(`#${id}`)?.value || "").trim();
      sourceText = v ? `${v.length} chars` : "Not provided";
    } else if (mode === "lora_training") {
      const files = $("#pipeline-lora-files")?.files;
      sourceText = files && files.length ? `${files.length} image(s)` : "No images";
    }
    $("#summary-source").textContent = sourceText;

    const phase = $("#pipeline-phase")?.value || "all";
    $("#summary-phase").textContent = {
      noop: "Smoke test", transcribe: "Transcribe", script_plan: "Script + Plan",
      render: "Render", assemble: "Assemble", all: "Full pipeline", lora_train: "LoRA Training",
    }[phase] || phase;

    const videoAdapter = selectedValue("video_adapter");
    const effectiveVideo = videoAdapter || (state.defaults?.video_adapter || "wan_s2v");
    $("#summary-video").textContent = VIDEO_LABELS[videoAdapter] || "Default";
    if (!videoAdapter && state.defaults?.video_adapter) {
      $("#summary-video").textContent = `Default (${VIDEO_LABELS[state.defaults.video_adapter] || state.defaults.video_adapter})`;
    }

    const lipsync = $("#pipeline-lipsync")?.value || "";
    $("#summary-lipsync").textContent = LIPSYNC_LABELS[lipsync] || "Default";
    if (!lipsync && state.defaults?.lipsync_adapter) {
      $("#summary-lipsync").textContent = `Default (${LIPSYNC_LABELS[state.defaults.lipsync_adapter] || state.defaults.lipsync_adapter})`;
    }

    const renderMode = selectedValue("render_mode");
    $("#summary-render").textContent = RENDER_LABELS[renderMode] || "Full";

    const lang = $("#pipeline-lang")?.value || "en";
    $("#summary-lang").textContent = LANG_LABELS[lang] || lang;

    // Output
    const parts = [];
    if ($("#pipeline-upscale")?.checked) parts.push("Upscale");
    if ($("#pipeline-enhance")?.checked) parts.push("Enhance");
    $("#summary-output").textContent = parts.length ? parts.join(" + ") : "Standard";

    // Intensity badge
    const pill = $("#pipeline-intensity-pill");
    if (pill) {
      if (GPU_PHASES.has(phase) || mode === "lora_training") {
        pill.textContent = "GPU intensive";
        pill.className = "animate-pill animate-pill-amber";
      } else if (phase === "noop") {
        pill.textContent = "No GPU";
        pill.className = "animate-pill animate-pill-gray";
      } else {
        pill.textContent = "Light GPU";
        pill.className = "animate-pill animate-pill-gray";
      }
    }

    // Wan Animate bridge
    const bridge = $("#pipeline-wan-animate-bridge");
    if (bridge) bridge.hidden = videoAdapter !== "wan_animate";
  }

  /* ── render: readiness checklist ─────────────────────────────────────── */

  function buildChecks() {
    const checks = [];
    const mode = currentSourceKind();
    const phase = $("#pipeline-phase")?.value || "all";
    const renderMode = selectedValue("render_mode");
    const videoAdapter = selectedValue("video_adapter");
    const effectiveVideo = videoAdapter || (state.defaults?.video_adapter || "wan_s2v");
    const lipsync = $("#pipeline-lipsync")?.value || "";
    const effectiveLipsync = lipsync || (state.defaults?.lipsync_adapter || "latentsync");
    const effectiveTts = state.defaults?.tts_adapter || "fish_s2";
    const isLora = mode === "lora_training";
    const needsGpu = GPU_PHASES.has(phase);
    const needsVoice = needsGpu && renderMode !== "source_audio";

    // 1. Source provided
    if (mode === "url") {
      const v = ($("#pipeline-url")?.value || "").trim();
      checks.push({ ready: !!v, label: "Source URL provided" });
    } else if (mode === "file") {
      const v = $("#pipeline-file")?.value || "";
      checks.push({ ready: !!v, label: "Local video selected" });
    } else if (mode === "story") {
      const v = ($("#pipeline-story-text")?.value || "").trim();
      checks.push({ ready: !!v, label: "Story text provided" });
    } else if (mode === "story_images") {
      const v = ($("#pipeline-story-images-text")?.value || "").trim();
      checks.push({ ready: !!v, label: "Story text provided" });
    } else if (isLora) {
      const member = $("#pipeline-lora-member")?.value;
      const files = $("#pipeline-lora-files")?.files;
      checks.push({ ready: !!member, label: "Cast member selected" });
      checks.push({ ready: files && files.length > 0, label: "Training images selected" });
    }

    // 2. Ollama (needed for any non-noop, non-lora phase)
    if (phase !== "noop" && !isLora && state.serviceReadiness) {
      const ollama = state.serviceReadiness.ollama;
      checks.push({
        ready: ollama?.ready ?? false,
        error: !(ollama?.ready),
        label: ollama?.ready ? "Ollama LLM is ready" : "Ollama LLM is unreachable",
      });
    }

    // 3. Video adapter (needed for render/all)
    if (needsGpu && effectiveVideo !== "wan_animate" && state.serviceReadiness?.video) {
      const vr = state.serviceReadiness.video[effectiveVideo];
      if (vr) {
        checks.push({
          ready: vr.ready,
          error: !vr.ready,
          label: vr.ready
            ? `${VIDEO_LABELS[effectiveVideo] || effectiveVideo} is ready`
            : `${VIDEO_LABELS[effectiveVideo] || effectiveVideo} is unreachable`,
        });
      }
    }

    // 4. TTS (needed for render/all, not source_audio)
    if (needsVoice && state.serviceReadiness?.tts) {
      const tr = state.serviceReadiness.tts[effectiveTts];
      if (tr) {
        checks.push({
          ready: tr.ready,
          error: !tr.ready,
          label: tr.ready
            ? `TTS (${effectiveTts === "fish_s2" ? "Fish S2" : "Chatterbox"}) is ready`
            : `TTS (${effectiveTts === "fish_s2" ? "Fish S2" : "Chatterbox"}) is unreachable`,
        });
      }
    }

    // 5. Lip-sync backend (if explicitly selected and not "none")
    if (needsGpu && effectiveLipsync !== "none" && state.serviceReadiness?.lipsync) {
      const lr = state.serviceReadiness.lipsync[effectiveLipsync];
      if (lr && effectiveVideo !== "wan_s2v") {
        checks.push({
          ready: lr.ready,
          label: lr.ready
            ? `${LIPSYNC_LABELS[effectiveLipsync] || effectiveLipsync} is ready`
            : `${LIPSYNC_LABELS[effectiveLipsync] || effectiveLipsync} is unreachable`,
        });
      }
    }

    // 6. LoRA files (needed for render/all)
    if (needsGpu && !isLora && state.currentCastMembers.length) {
      const allLora = state.currentCastMembers.every((m) => m.has_lora);
      checks.push({
        ready: allLora,
        error: !allLora,
        label: allLora ? "Cast LoRA files present" : "Cast LoRA files missing (Track B)",
      });
    }

    // 7. Voice files (needed for render/all, not source_audio)
    if (needsVoice && !isLora && state.currentCastMembers.length) {
      const allVoice = state.currentCastMembers.every((m) => m.has_voice);
      checks.push({
        ready: allVoice,
        error: !allVoice,
        label: allVoice ? "Voice reference files present" : "Voice reference files missing (Track B)",
      });
    }

    // 8. Rights
    if (phase !== "noop") {
      const rightsOk = !!$("#pipeline-rights-cleared")?.checked;
      checks.push({ ready: rightsOk, label: rightsOk ? "Usage rights confirmed" : "Usage rights not confirmed" });
    }

    return checks;
  }

  function renderReadiness() {
    const checks = buildChecks();
    const container = $("#pipeline-readiness-list");
    container.replaceChildren(...checks.map((check) => {
      const row = document.createElement("div");
      row.className = `animate-check${check.ready ? " ready" : ""}${check.error ? " error" : ""}`;
      row.textContent = check.label;
      return row;
    }));

    const allReady = checks.every((c) => c.ready) && !state.submitting;
    $("#pipeline-submit").disabled = !allReady;

    // Header readiness badge
    const badge = $("#pipeline-readiness");
    const label = $("#pipeline-readiness-label");
    if (!state.serviceReadiness) {
      badge.className = "animate-readiness";
      label.textContent = "Checking readiness…";
    } else if (allReady) {
      badge.className = "animate-readiness ready";
      label.textContent = "Ready";
    } else {
      badge.className = "animate-readiness blocked";
      label.textContent = "Not ready";
    }
  }

  /* ── render: visibility toggles ─────────────────────────────────────── */

  function renderVisibility() {
    const phase = $("#pipeline-phase")?.value || "all";
    const phaseHint = $("#pipeline-phase-hint");
    if (phaseHint) phaseHint.textContent = PHASE_HINTS[phase] || "";
  }

  /* ── renderAll ──────────────────────────────────────────────────────── */

  function renderAll() {
    renderSummary();
    renderReadiness();
    renderVisibility();
  }

  /* ── submit ─────────────────────────────────────────────────────────── */

  async function submitJob(e) {
    e.preventDefault();
    const errEl = $("#pipeline-submit-error");
    errEl.hidden = true;
    const btn = $("#pipeline-submit");

    if (selectedValue("video_adapter") === "wan_animate") {
      window.location.assign("/animate/new");
      return;
    }

    state.submitting = true;
    btn.disabled = true;
    btn.textContent = "Submitting…";
    renderAll();

    try {
      const sourceKind = currentSourceKind();
      let sourceUrl = "";
      let storyText = null;
      const characterImages = {};
      let loraTraining = null;

      if (sourceKind === "url") {
        sourceUrl = ($("#pipeline-url")?.value || "").trim();
        if (!sourceUrl) throw new Error("Source URL is required.");
      } else if (sourceKind === "file") {
        sourceUrl = $("#pipeline-file")?.value || "";
        if (!sourceUrl) throw new Error("Select a local video file.");
      } else if (sourceKind === "story") {
        storyText = ($("#pipeline-story-text")?.value || "").trim();
        if (!storyText) throw new Error("Story text is required.");
      } else if (sourceKind === "story_images") {
        storyText = ($("#pipeline-story-images-text")?.value || "").trim();
        if (!storyText) throw new Error("Story text is required.");
      } else if (sourceKind === "lora_training") {
        const memberId = $("#pipeline-lora-member")?.value;
        const files = Array.from($("#pipeline-lora-files")?.files || []);
        if (!memberId) throw new Error("Select a cast member.");
        if (!files.length) throw new Error("Choose at least one LoRA training image.");
        const statusEl = $("#pipeline-lora-upload-status");
        const paths = [];
        for (let i = 0; i < files.length; i++) {
          if (statusEl) statusEl.textContent = `Uploading ${i + 1}/${files.length}…`;
          const caption = $(`#pipeline-lora-caption-${i}`)?.value?.trim() || "";
          paths.push(await uploadLoraTrainingImage(memberId, files[i], caption));
        }
        if (statusEl) statusEl.textContent = `${paths.length} image(s) staged.`;
        sourceUrl = "lora-training://dashboard-upload";
        loraTraining = { cast_member_id: memberId, image_paths: paths };
      }

      // Story+images character uploads
      if (sourceKind === "story_images") {
        const inputs = $$("#pipeline-images-slots input[type=file]");
        let hasAny = false;
        for (const inp of inputs) {
          if (inp.files.length) {
            const result = await uploadCharacterImage(inp.dataset.memberId);
            if (result) {
              characterImages[result.member_id] = result.path;
              hasAny = true;
            }
          }
        }
        if (!hasAny) throw new Error("Upload at least one character image.");
      }

      const videoAdapter = selectedValue("video_adapter");
      const lipsyncAdapter = $("#pipeline-lipsync")?.value || "";
      const whisperLanguage = $("#pipeline-whisper-lang")?.value || "";
      const hasRealAudio = !(sourceKind === "story" || sourceKind === "story_images" || sourceKind === "lora_training");
      const maxShotDur = parseFloat($("#pipeline-max-shot-dur")?.value || "8");
      const gpuPrice = parseFloat($("#pipeline-gpu-price")?.value || "0");
      const castSel = $("#pipeline-cast");
      const renderMode = selectedValue("render_mode");
      const audioProfile = $("#pipeline-audio-profile")?.value || "auto";

      const overrides = {};
      if (videoAdapter) overrides.video_adapter = videoAdapter;
      if (lipsyncAdapter) overrides.lipsync_adapter = lipsyncAdapter;
      overrides.video_upscale_enabled = !!$("#pipeline-upscale")?.checked;
      overrides.video_enhance_enabled = !!$("#pipeline-enhance")?.checked;
      overrides.video_enhance_adapter = $("#pipeline-enhance-adapter")?.value || "ffmpeg";
      if (hasRealAudio && whisperLanguage) overrides.whisper_language = whisperLanguage;
      const isolateCb = $("#pipeline-isolate-vocals");
      if (isolateCb && !isolateCb.disabled && isolateCb.checked) {
        overrides.whisper_isolate_vocals = true;
      }
      if (maxShotDur && maxShotDur !== 8) overrides.max_shot_duration_sec = maxShotDur;
      if ($("#pipeline-auto-plan")?.checked) overrides.auto_approve_plan = true;
      if ($("#pipeline-auto-images")?.checked) overrides.auto_approve_images = true;
      if ($("#pipeline-auto-transcript")?.checked) overrides.auto_approve_transcript = true;

      const body = {
        source: { kind: sourceKind, url: sourceUrl },
        rights_cleared: !!$("#pipeline-rights-cleared")?.checked,
        target_language: $("#pipeline-lang")?.value || "en",
        phase: $("#pipeline-phase")?.value || "all",
        render_mode: renderMode,
        audio_profile: audioProfile,
        gpu_price_per_hour: Number.isFinite(gpuPrice) ? gpuPrice : 0,
        overrides: overrides,
      };
      if (castSel && castSel.value) body.cast_ref = castSel.value;
      if (storyText) body.story_text = storyText;
      if (Object.keys(characterImages).length) body.character_images = characterImages;
      if (loraTraining) body.lora_training = loraTraining;

      const data = await requestJson(URLS.create, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      window.location.href = "/jobs/" + data.job_id;
    } catch (err) {
      errEl.textContent = err.message;
      errEl.hidden = false;
      state.submitting = false;
      btn.disabled = false;
      btn.textContent = "Create pipeline job";
      renderAll();
    }
  }

  /* ── event binding ──────────────────────────────────────────────────── */

  function bindEvents() {
    // Source kind radio cards
    $$('input[name="source_kind"]').forEach((inp) => {
      inp.addEventListener("change", () => toggleSourceMode(inp.value));
    });

    // Cast change
    const castSel = $("#pipeline-cast");
    if (castSel) castSel.addEventListener("change", () => onCastChange(castSel.value));

    // Phase change
    const phaseSel = $("#pipeline-phase");
    if (phaseSel) phaseSel.addEventListener("change", () => renderAll());

    // Video adapter radio cards
    $$('input[name="video_adapter"]').forEach((inp) => {
      inp.addEventListener("change", () => renderAll());
    });

    // Render mode segmented control
    $$('input[name="render_mode"]').forEach((inp) => {
      inp.addEventListener("change", () => renderAll());
    });

    // Lip-sync select
    const lipsyncSel = $("#pipeline-lipsync");
    if (lipsyncSel) lipsyncSel.addEventListener("change", () => renderAll());

    // Language
    const langSel = $("#pipeline-lang");
    if (langSel) langSel.addEventListener("change", () => renderAll());

    // Source inputs — update summary on input
    const urlInput = $("#pipeline-url");
    if (urlInput) urlInput.addEventListener("input", () => renderAll());

    const fileSel = $("#pipeline-file");
    if (fileSel) fileSel.addEventListener("change", () => renderAll());

    const storyText = $("#pipeline-story-text");
    if (storyText) storyText.addEventListener("input", () => renderAll());

    const storyImagesText = $("#pipeline-story-images-text");
    if (storyImagesText) storyImagesText.addEventListener("input", () => renderAll());

    // File browser refresh
    const refreshBtn = $("#pipeline-refresh-files");
    if (refreshBtn) refreshBtn.addEventListener("click", loadLocalVideos);

    const fileDir = $("#pipeline-file-dir");
    if (fileDir) {
      fileDir.addEventListener("keydown", (e) => {
        if (e.key === "Enter") { e.preventDefault(); loadLocalVideos(); }
      });
      fileDir.addEventListener("blur", loadLocalVideos);
    }

    // LoRA files change
    const loraFiles = $("#pipeline-lora-files");
    if (loraFiles) loraFiles.addEventListener("change", rebuildLoraCaptionRows);

    const loraMember = $("#pipeline-lora-member");
    if (loraMember) loraMember.addEventListener("change", () => {
      rebuildLoraCaptionRows();
      renderAll();
    });

    // Audio profile
    const audioProfile = $("#pipeline-audio-profile");
    if (audioProfile) audioProfile.addEventListener("change", () => {
      updateIsolateVocals();
      renderAll();
    });

    // Checkboxes that affect summary/readiness
    ["pipeline-upscale", "pipeline-enhance", "pipeline-rights-cleared",
     "pipeline-auto-plan", "pipeline-auto-images", "pipeline-auto-transcript",
     "pipeline-isolate-vocals"].forEach((id) => {
      const el = $(`#${id}`);
      if (el) el.addEventListener("change", () => renderAll());
    });

    // Enhance adapter
    const enhanceAdapter = $("#pipeline-enhance-adapter");
    if (enhanceAdapter) enhanceAdapter.addEventListener("change", () => renderAll());

    // Submit
    const form = $("#pipeline-job-form");
    if (form) form.addEventListener("submit", submitJob);
  }

  /* ── init ────────────────────────────────────────────────────────────── */

  bindEvents();
  renderAll();
  loadOptions();
})();
