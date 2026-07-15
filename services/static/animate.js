/* Animate Studio — dependency-free controller for direct Wan 2.2 Animate jobs. */

(function () {
  "use strict";

  const root = document.getElementById("animate-studio");
  if (!root) return;

  const $ = (selector, scope = document) => scope.querySelector(selector);
  const $$ = (selector, scope = document) => Array.from(scope.querySelectorAll(selector));
  const selectedValue = (name) => $(`input[name="${name}"]:checked`)?.value || "";

  const URLS = {
    options: root.dataset.optionsUrl || "/api/animate/options",
    create: root.dataset.createUrl || "/api/jobs",
    uploadVideo: "/api/assets/video/upload",
    importVideo: "/api/assets/video/from-url",
    serverFiles: "/api/assets/video/server-files",
    importServerVideo: "/api/assets/video/from-server-file",
    uploadImage: "/api/assets/image/upload",
  };

  const state = {
    options: null,
    casts: [],
    backendReady: null,
    backendReason: "",
    fluxEditEnabled: null,
    fluxEditReason: "Checking FLUX.2 reference-edit availability…",
    fluxEditMaxUserReferences: 3,
    wanFluxRetargetEnabled: null,
    wanFluxRetargetReason: "Checking optional FLUX.1 Kontext retargeting…",
    maxDriverRangeSec: 30,
    lipsyncReadiness: {},
    driverSource: "url",
    driver: null,
    driverPreviewObjectUrl: null,
    serverFiles: new Map(),
    garmentAssets: [],
    accessoryAssets: [],
    exactImage: null,
    pendingUploads: 0,
    driverBusy: false,
    submitting: false,
  };

  function coalesce(...values) {
    return values.find((value) => value !== undefined && value !== null);
  }

  function asNumber(value) {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : null;
  }

  function formatBytes(bytes) {
    const value = asNumber(bytes);
    if (value === null) return "";
    if (value < 1024 * 1024) return `${(value / 1024).toFixed(0)} KiB`;
    if (value < 1024 * 1024 * 1024) return `${(value / 1024 / 1024).toFixed(1)} MiB`;
    return `${(value / 1024 / 1024 / 1024).toFixed(2)} GiB`;
  }

  function formatDuration(seconds) {
    const value = asNumber(seconds);
    if (value === null) return "";
    if (value < 60) return `${value.toFixed(1)}s`;
    const minutes = Math.floor(value / 60);
    return `${minutes}m ${(value % 60).toFixed(0)}s`;
  }

  function errorMessage(payload, fallback) {
    const detail = payload?.detail;
    if (Array.isArray(detail)) {
      return detail.map((item) => {
        const path = Array.isArray(item.loc) ? item.loc.slice(1).join(".") : "";
        return path ? `${path}: ${item.msg}` : item.msg;
      }).join("\n");
    }
    return detail?.message || detail || payload?.message || fallback;
  }

  async function requestJson(url, options = {}) {
    const response = await fetch(url, options);
    let payload = {};
    try {
      payload = await response.json();
    } catch (_) {
      payload = {};
    }
    if (!response.ok) throw new Error(errorMessage(payload, `Request failed (${response.status})`));
    return payload;
  }

  function xhrForm(url, formData, onProgress) {
    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open("POST", url);
      xhr.responseType = "json";
      xhr.upload.addEventListener("progress", (event) => {
        if (event.lengthComputable && onProgress) onProgress(event.loaded / event.total);
      });
      xhr.addEventListener("load", () => {
        const payload = xhr.response || {};
        if (xhr.status >= 200 && xhr.status < 300) resolve(payload);
        else reject(new Error(errorMessage(payload, `Upload failed (${xhr.status})`)));
      });
      xhr.addEventListener("error", () => reject(new Error("Upload failed because the network connection was interrupted.")));
      xhr.addEventListener("abort", () => reject(new Error("Upload was cancelled.")));
      xhr.send(formData);
    });
  }

  function normalizeAsset(payload, fallbackName = "Asset") {
    const raw = payload?.asset || payload;
    const metadata = raw?.metadata || {};
    const assetId = coalesce(raw?.asset_id, raw?.id);
    if (!assetId) throw new Error("The server did not return an asset ID.");
    return {
      asset_id: assetId,
      name: coalesce(raw.original_name, raw.filename, raw.name, fallbackName),
      mime_type: raw.mime_type || "",
      size_bytes: coalesce(raw.size_bytes, metadata.size_bytes),
      metadata,
      media_url: coalesce(raw.media_url, raw.preview_url, `/api/assets/${encodeURIComponent(assetId)}/media`),
    };
  }

  function setError(id, message) {
    const element = document.getElementById(id);
    if (!element) return;
    element.textContent = message || "";
    element.hidden = !message;
  }

  function setProgress(percent, label, indeterminate = false) {
    const wrapper = $("#animate-driver-progress");
    const bar = $("#animate-driver-progress-bar");
    wrapper.hidden = false;
    wrapper.classList.toggle("indeterminate", indeterminate);
    const value = Math.max(0, Math.min(100, Math.round(percent || 0)));
    bar.style.width = indeterminate ? "" : `${value}%`;
    wrapper.setAttribute("aria-valuenow", indeterminate ? "" : String(value));
    $("#animate-driver-progress-label").textContent = label;
  }

  function hideProgress() {
    $("#animate-driver-progress").hidden = true;
    $("#animate-driver-progress").classList.remove("indeterminate");
  }

  function chooseDriverTab(source) {
    state.driverSource = source;
    $$('[data-driver-tab]').forEach((button) => {
      const active = button.dataset.driverTab === source;
      button.classList.toggle("active", active);
      button.setAttribute("aria-selected", active ? "true" : "false");
      $(`#driver-panel-${button.dataset.driverTab}`).hidden = !active;
    });
    setError("animate-driver-error", "");
  }

  function videoMetadata(asset) {
    const metadata = asset?.metadata || {};
    return {
      duration: asNumber(coalesce(metadata.duration_sec, metadata.duration, asset?.duration_sec)),
      width: asNumber(coalesce(metadata.width, metadata.video_width)),
      height: asNumber(coalesce(metadata.height, metadata.video_height)),
      fps: asNumber(coalesce(metadata.fps, metadata.frame_rate)),
      codec: coalesce(metadata.codec, metadata.video_codec, ""),
      hasAudio: coalesce(metadata.has_audio, metadata.audio_stream, metadata.audio_streams ? metadata.audio_streams > 0 : null),
      chunkCount: asNumber(coalesce(metadata.estimated_chunk_count, metadata.chunk_count)),
    };
  }

  function setDriver(asset, previewUrl = "") {
    if (state.driverPreviewObjectUrl && state.driverPreviewObjectUrl !== previewUrl) {
      URL.revokeObjectURL(state.driverPreviewObjectUrl);
      state.driverPreviewObjectUrl = null;
    }
    state.driver = asset;
    const video = $("#animate-driver-preview");
    video.src = previewUrl || asset.media_url;
    video.load();
    $("#animate-driver-name").textContent = asset.name || "Driving video";
    renderDriverMetadata();
    $("#animate-driver-preview-card").hidden = false;
    $("#animate-long-driver-ack").checked = false;
    $("#animate-target-confirmed").checked = false;
    setError("animate-driver-error", "");
    renderAll();
  }

  function removeDriver() {
    if (state.driverPreviewObjectUrl) URL.revokeObjectURL(state.driverPreviewObjectUrl);
    state.driverPreviewObjectUrl = null;
    state.driver = null;
    const video = $("#animate-driver-preview");
    video.pause();
    video.removeAttribute("src");
    video.load();
    $("#animate-driver-preview-card").hidden = true;
    $("#animate-long-driver-confirm").hidden = true;
    $("#animate-long-driver-ack").checked = false;
    $("#animate-target-confirmed").checked = false;
    $("#animate-driver-file").value = "";
    renderAll();
  }

  function renderDriverMetadata() {
    if (!state.driver) return;
    const metadata = videoMetadata(state.driver);
    const values = [];
    if (metadata.duration !== null) values.push(formatDuration(metadata.duration));
    if (metadata.width && metadata.height) values.push(`${metadata.width}×${metadata.height}`);
    if (metadata.fps) values.push(`${metadata.fps.toFixed(metadata.fps % 1 ? 2 : 0)} FPS`);
    if (metadata.codec) values.push(String(metadata.codec).toUpperCase());
    if (state.driver.size_bytes) values.push(formatBytes(state.driver.size_bytes));
    if (metadata.hasAudio === true) values.push("Audio detected");
    if (metadata.hasAudio === false) values.push("Silent");

    const container = $("#animate-driver-metadata");
    container.replaceChildren(...values.map((value) => {
      const span = document.createElement("span");
      span.textContent = value;
      return span;
    }));

    const chunks = metadata.chunkCount || (metadata.duration ? Math.ceil(metadata.duration / (77 / 30)) : null);
    $("#animate-driver-guidance").textContent = chunks && chunks > 1
      ? `Estimated ${chunks} internal generation chunks. Continuity is checked before publishing.`
      : "The server normalizes the driver to constant 30 FPS before generation.";
  }

  function effectiveDuration() {
    if (!state.driver) return null;
    if ($("#animate-timeline").value === "selected_range") {
      const start = asNumber($("#animate-start-sec").value) || 0;
      const end = asNumber($("#animate-end-sec").value);
      if (end !== null && end > start) return end - start;
    }
    return videoMetadata(state.driver).duration;
  }

  async function uploadDriver(file) {
    if (!file) return;
    setError("animate-driver-error", "");
    if (file.size > 2 * 1024 * 1024 * 1024) {
      setError("animate-driver-error", "The driving video exceeds the 2 GiB upload limit.");
      return;
    }

    if (state.driverPreviewObjectUrl) URL.revokeObjectURL(state.driverPreviewObjectUrl);
    state.driverPreviewObjectUrl = URL.createObjectURL(file);
    const preview = $("#animate-driver-preview");
    preview.src = state.driverPreviewObjectUrl;
    $("#animate-driver-name").textContent = file.name;
    $("#animate-driver-preview-card").hidden = false;

    state.driverBusy = true;
    setProgress(0, `Uploading ${file.name}…`);
    renderAll();
    try {
      const form = new FormData();
      form.append("file", file);
      const payload = await xhrForm(URLS.uploadVideo, form, (fraction) => {
        setProgress(fraction * 100, `Uploading ${Math.round(fraction * 100)}%`);
      });
      setProgress(100, "Validating video…", true);
      const asset = normalizeAsset(payload, file.name);
      setDriver(asset, state.driverPreviewObjectUrl);
    } catch (error) {
      const message = error.message;
      removeDriver();
      setError("animate-driver-error", message);
    } finally {
      state.driverBusy = false;
      hideProgress();
      renderAll();
    }
  }

  async function importDriverUrl() {
    const value = $("#animate-driver-url").value.trim();
    setError("animate-driver-error", "");
    let url;
    try {
      url = new URL(value);
      if (!['http:', 'https:'].includes(url.protocol)) throw new Error();
    } catch (_) {
      setError("animate-driver-error", "Enter a valid HTTP(S) video URL.");
      return;
    }

    state.driverBusy = true;
    $("#animate-import-url").disabled = true;
    setProgress(0, "Downloading and inspecting video…", true);
    renderAll();
    try {
      const payload = await requestJson(URLS.importVideo, {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({url: value}),
      });
      setDriver(normalizeAsset(payload, url.hostname));
    } catch (error) {
      setError("animate-driver-error", error.message);
    } finally {
      state.driverBusy = false;
      $("#animate-import-url").disabled = false;
      hideProgress();
      renderAll();
    }
  }

  async function loadServerFiles() {
    const select = $("#animate-server-file");
    const useButton = $("#animate-use-server-file");
    select.disabled = true;
    useButton.disabled = true;
    select.replaceChildren(new Option("Loading server files…", ""));
    try {
      const payload = await requestJson(URLS.serverFiles);
      const files = payload.files || payload.videos || [];
      state.serverFiles.clear();
      select.replaceChildren(new Option(files.length ? "Choose a server video" : "No server videos found", ""));
      files.forEach((file, index) => {
        const token = String(coalesce(file.file_id, file.id, file.token, index));
        state.serverFiles.set(token, file);
        const size = file.size_bytes ? ` · ${formatBytes(file.size_bytes)}` : "";
        select.add(new Option(`${file.name || file.original_name || "Video"}${size}`, token));
      });
      select.disabled = files.length === 0;
    } catch (error) {
      select.replaceChildren(new Option("Unable to load server files", ""));
      setError("animate-driver-error", error.message);
    }
  }

  async function importServerFile() {
    const token = $("#animate-server-file").value;
    if (!token) {
      setError("animate-driver-error", "Choose a server video first.");
      return;
    }
    state.driverBusy = true;
    $("#animate-use-server-file").disabled = true;
    setProgress(0, "Staging and inspecting server video…", true);
    renderAll();
    try {
      const payload = await requestJson(URLS.importServerVideo, {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({file_id: token}),
      });
      const file = state.serverFiles.get(token);
      setDriver(normalizeAsset(payload, file?.name || "Server video"));
    } catch (error) {
      setError("animate-driver-error", error.message);
    } finally {
      state.driverBusy = false;
      $("#animate-use-server-file").disabled = !$("#animate-server-file").value;
      hideProgress();
      renderAll();
    }
  }

  function normalizeCasts(payload) {
    const casts = payload.casts || [];
    return casts.map((cast) => ({
      ...cast,
      id: cast.id || cast.cast_ref,
      name: cast.name || cast.label || cast.id || cast.cast_ref,
      members: (cast.members || []).map((member) => ({
        ...member,
        id: member.id || member.member_id,
        name: member.name || member.label || member.id || member.member_id,
        has_lora: coalesce(member.has_lora, member.lora_available, member.capabilities?.lora),
        has_voice: coalesce(member.has_voice, member.voice_available, member.capabilities?.voice),
      })),
    }));
  }

  function deriveReadiness(options) {
    const readiness = options.readiness || {};
    const capabilities = options.capabilities || {};
    const animateCapability = capabilities.wan_animate;
    state.backendReady = coalesce(
      readiness.ready,
      readiness.wan_animate_ready,
      readiness.wan_animate?.ready,
      capabilities.wan_animate_ready,
      typeof animateCapability === "boolean" ? animateCapability : animateCapability?.ready,
      null,
    );
    state.backendReason = coalesce(
      readiness.reason,
      readiness.wan_animate?.reason,
      animateCapability?.reason,
      "",
    );
    state.fluxEditEnabled = coalesce(options.features?.flux2_edit_enabled, null);
    state.fluxEditMaxUserReferences = Math.max(
      0,
      Number(options.features?.flux2_edit_max_user_references ?? 3) || 0,
    );
    state.fluxEditReason = coalesce(
      options.features?.flux2_edit_reason,
      options.features?.flux2_edit_unavailable_reason,
      "Image-guided complete-look references are unavailable. Text-directed styling still works.",
    );
    state.wanFluxRetargetEnabled = coalesce(
      options.features?.wan_flux_retarget_enabled,
      false,
    );
    state.wanFluxRetargetReason = coalesce(
      options.features?.wan_flux_retarget_reason,
      "Optional FLUX.1 Kontext retargeting is not installed.",
    );
    state.lipsyncReadiness = readiness.lipsync || {};
    state.maxDriverRangeSec = Number(options.limits?.max_driver_range_sec) || 30;
  }

  function renderFeatureAvailability() {
    const unavailable = state.fluxEditEnabled !== true || state.fluxEditMaxUserReferences < 1;
    ["garment", "accessory"].forEach((kind) => {
      const input = $(`#wardrobe-${kind}-files`);
      const button = $(`#wardrobe-${kind}-button`);
      input.disabled = unavailable;
      button.classList.toggle("is-disabled", unavailable);
      button.setAttribute("aria-disabled", unavailable ? "true" : "false");
      button.title = unavailable ? state.fluxEditReason : "";
    });
    $("#animate-flux2-edit-note").hidden = !unavailable;
    $("#animate-flux2-edit-message").textContent = state.fluxEditReason;
    const max = state.fluxEditMaxUserReferences;
    $("#wardrobe-garment-hint").textContent = unavailable
      ? "Text-directed design remains available."
      : `Optional · up to ${max} clothing/styling references combined.`;
    $("#wardrobe-accessory-hint").textContent = unavailable
      ? "Image-guided styling details are currently unavailable."
      : `Jewelry, bags, footwear, makeup, or other details · shares the ${max}-image limit.`;

    const backendSelect = $("#animate-lipsync-backend");
    Array.from(backendSelect.options).forEach((option) => {
      option.disabled = state.lipsyncReadiness[option.value]?.ready === false;
    });
    if (backendSelect.selectedOptions[0]?.disabled) {
      const readyOption = Array.from(backendSelect.options).find((option) => !option.disabled);
      if (readyOption) backendSelect.value = readyOption.value;
    }
  }

  async function loadOptions() {
    let options;
    try {
      options = await requestJson(URLS.options);
    } catch (_) {
      // A casts fallback keeps the page useful while a partially deployed API is upgraded.
      try {
        options = await requestJson("/api/casts");
      } catch (error) {
        options = {casts: []};
        state.backendReady = false;
        state.backendReason = error.message;
      }
    }
    state.options = options;
    state.casts = normalizeCasts(options);
    if (state.backendReady !== false) deriveReadiness(options);
    populateCasts(options.default || options.defaults?.cast_ref);
    renderAll();
  }

  function populateCasts(defaultCast) {
    const select = $("#animate-cast");
    select.replaceChildren(new Option(state.casts.length ? "Choose a cast" : "No casts available", ""));
    state.casts.forEach((cast) => select.add(new Option(cast.name || cast.id, cast.id)));
    select.disabled = state.casts.length === 0;
    if (defaultCast && state.casts.some((cast) => cast.id === defaultCast)) select.value = defaultCast;
    else if (state.casts.length === 1) select.value = state.casts[0].id;
    populateMembers();
  }

  function selectedCast() {
    return state.casts.find((cast) => cast.id === $("#animate-cast").value) || null;
  }

  function selectedMember() {
    return selectedCast()?.members.find((member) => member.id === $("#animate-member").value) || null;
  }

  function populateMembers(previousMember = "") {
    const select = $("#animate-member");
    const cast = selectedCast();
    const members = cast?.members || [];
    select.replaceChildren(new Option(members.length ? "Choose a target member" : "Choose a cast first", ""));
    members.forEach((member) => select.add(new Option(member.name, member.id)));
    select.disabled = members.length === 0;
    if (previousMember && members.some((member) => member.id === previousMember)) select.value = previousMember;
    else if (members.length === 1) select.value = members[0].id;
    setError("animate-target-error", "");
    renderMemberCapabilities();
    renderAll();
  }

  function renderMemberCapabilities() {
    const member = selectedMember();
    const parts = [];
    if (member?.has_lora === true) parts.push("LoRA ready");
    if (member?.has_lora === false) parts.push("LoRA unavailable");
    if (member?.has_voice === true) parts.push("Voice ready");
    if (member?.has_voice === false) parts.push("Voice unavailable");
    $("#animate-member-capabilities").textContent = parts.join(" · ");
  }

  function makeImageItem(file, kind) {
    return {
      local_id: `${kind}-${Date.now()}-${Math.random().toString(16).slice(2)}`,
      file,
      name: file.name,
      preview_url: URL.createObjectURL(file),
      asset_id: null,
      status: "uploading",
      error: "",
    };
  }

  function renderImageCollection(kind) {
    const items = kind === "garment" ? state.garmentAssets : state.accessoryAssets;
    const container = $(`#wardrobe-${kind}-list`);
    container.replaceChildren(...items.map((item) => {
      const row = document.createElement("div");
      row.className = "animate-thumbnail";
      const image = document.createElement("img");
      image.src = item.preview_url;
      image.alt = "";
      const details = document.createElement("div");
      details.style.minWidth = "0";
      const name = document.createElement("div");
      name.className = "animate-thumbnail-name";
      name.textContent = item.name;
      const status = document.createElement("div");
      status.className = `animate-thumbnail-state ${item.status === "ready" ? "ready" : item.status === "failed" ? "failed" : ""}`;
      status.textContent = item.status === "ready" ? "Uploaded" : item.status === "failed" ? item.error : "Uploading…";
      details.append(name, status);
      const remove = document.createElement("button");
      remove.type = "button";
      remove.className = "animate-thumbnail-remove";
      remove.dataset.removeImage = item.local_id;
      remove.dataset.imageKind = kind;
      remove.setAttribute("aria-label", `Remove ${item.name}`);
      remove.textContent = "×";
      row.append(image, details, remove);
      return row;
    }));
  }

  async function addReferenceImages(files, kind) {
    if (state.fluxEditEnabled !== true) return;
    const collection = kind === "garment" ? state.garmentAssets : state.accessoryAssets;
    const requested = Array.from(files || []);
    const currentTotal = state.garmentAssets.length + state.accessoryAssets.length;
    const slots = Math.max(0, state.fluxEditMaxUserReferences - currentTotal);
    const allowed = requested.slice(0, slots);
    if (requested.length > allowed.length) {
      setError(
        "animate-wardrobe-error",
        `FLUX.2 accepts up to ${state.fluxEditMaxUserReferences} clothing/styling references combined.`,
      );
    }
    if (!allowed.length) {
      renderAll();
      return;
    }
    const items = allowed.map((file) => makeImageItem(file, kind));
    collection.push(...items);
    renderImageCollection(kind);
    state.pendingUploads += items.length;
    renderAll();

    await Promise.all(items.map(async (item) => {
      try {
        const form = new FormData();
        form.append("file", item.file);
        form.append("purpose", kind);
        const payload = await xhrForm(URLS.uploadImage, form);
        item.asset_id = normalizeAsset(payload, item.name).asset_id;
        item.status = "ready";
      } catch (error) {
        item.status = "failed";
        item.error = error.message;
      } finally {
        state.pendingUploads -= 1;
        renderImageCollection(kind);
        renderAll();
      }
    }));
  }

  function removeReferenceImage(localId, kind) {
    const key = kind === "garment" ? "garmentAssets" : "accessoryAssets";
    const index = state[key].findIndex((item) => item.local_id === localId);
    if (index < 0) return;
    URL.revokeObjectURL(state[key][index].preview_url);
    state[key].splice(index, 1);
    setError("animate-wardrobe-error", "");
    renderImageCollection(kind);
    renderAll();
  }

  async function uploadExactImage(file) {
    if (!file) return;
    setError("animate-character-error", "");
    if (state.exactImage?.preview_url) URL.revokeObjectURL(state.exactImage.preview_url);
    const item = makeImageItem(file, "character");
    state.exactImage = item;
    const preview = $("#animate-character-preview");
    preview.src = item.preview_url;
    preview.hidden = false;
    $("#animate-character-placeholder").hidden = true;
    $("#animate-character-upload-status").textContent = `Uploading ${file.name}…`;
    state.pendingUploads += 1;
    renderAll();
    try {
      const form = new FormData();
      form.append("file", file);
      form.append("purpose", "character");
      const payload = await xhrForm(URLS.uploadImage, form, (fraction) => {
        $("#animate-character-upload-status").textContent = `Uploading ${Math.round(fraction * 100)}% · ${file.name}`;
      });
      item.asset_id = normalizeAsset(payload, file.name).asset_id;
      item.status = "ready";
      $("#animate-character-upload-status").textContent = `${file.name} · uploaded`;
    } catch (error) {
      item.status = "failed";
      item.error = error.message;
      setError("animate-character-error", error.message);
      $("#animate-character-upload-status").textContent = file.name;
    } finally {
      state.pendingUploads -= 1;
      renderAll();
    }
  }

  function setModePanels() {
    const replace = selectedValue("animate_mode") === "replace";
    $("#animate-motion-advanced").hidden = replace;
    $("#animate-replace-advanced").hidden = !replace;
    if (replace) {
      $("#animate-retarget-pose").checked = false;
      $("#animate-flux-retarget").checked = false;
      $("#animate-flux-retarget-row").hidden = true;
    }
  }

  function setLookPanel() {
    const look = selectedValue("look_source");
    $("#animate-look-auto").hidden = look !== "auto_lora";
    $("#animate-look-styled").hidden = look !== "styled_lora";
    $("#animate-look-exact").hidden = look !== "exact_image";
  }

  function applyAudioDefaults() {
    const mode = selectedValue("audio_mode");
    const enabled = $("#animate-lipsync-enabled");
    const hint = $("#animate-audio-hint");
    if (mode === "cast_voice") {
      enabled.checked = true;
      $("#animate-lipsync-backend").value = "latentsync";
      hint.textContent = "Transcribes the driver and re-synthesizes the same words with the selected member’s voice, matched to driver timing.";
    } else if (mode === "none") {
      enabled.checked = false;
      hint.textContent = "Exports the animation without an audio track.";
    } else {
      enabled.checked = false;
      hint.textContent = "Preserves the driving video’s audio and timing.";
    }
    renderLipSync();
  }

  function renderLipSync() {
    const toggle = $("#animate-lipsync-enabled");
    const audio = selectedValue("audio_mode");
    toggle.disabled = audio === "none";
    if (audio === "none") toggle.checked = false;
    const enabled = toggle.checked;
    const backend = $("#animate-lipsync-backend").value;
    const backendReadiness = state.lipsyncReadiness[backend];
    $("#animate-lipsync-backend-row").hidden = !enabled;
    $("#animate-lipsync-hint").textContent = enabled
      ? backendReadiness?.ready === false
        ? backendReadiness.reason || "The selected lip-sync service is unavailable."
        : "Runs only after Wan unloads from the GPU."
      : audio === "cast_voice"
        ? "Recommended for newly synthesized cast speech."
        : "Off by default for transferred source audio to protect identity.";
  }

  function selectedStyleTargets() {
    return $$("input[name=\"style_change_target\"]:checked").map((input) => input.value);
  }

  function wardrobeHasDirection() {
    return selectedStyleTargets().length > 0;
  }

  function completeLookScopeErrors() {
    if (!wardrobeHasDirection()) {
      return ["Choose at least one styling category that FLUX.2 may change."];
    }
    const selected = new Set(selectedStyleTargets());
    const errors = [];
    const scopedFields = [
      ["wardrobe-clothing-type", "clothing", "Clothing / dress"],
      ["wardrobe-primary-color", "clothing", "Clothing / dress"],
      ["wardrobe-material-pattern", "clothing", "Clothing / dress"],
      ["wardrobe-jewelry", "jewelry", "Jewelry"],
      ["wardrobe-bags", "bags", "Bags"],
      ["wardrobe-footwear", "footwear", "Footwear"],
      ["wardrobe-makeup", "makeup", "Makeup / lipstick"],
      ["wardrobe-hair", "hair", "Hair"],
      ["wardrobe-accessories", "other", "Other styling"],
    ];
    scopedFields.forEach(([id, target, label]) => {
      if ($(`#${id}`).value.trim() && !selected.has(target)) {
        errors.push(`Select ${label} under “What should FLUX.2 change?” or clear its field.`);
      }
    });
    const hasClothingReference = state.garmentAssets.some((item) => item.status === "ready");
    if (hasClothingReference && !selected.has("clothing")) {
      errors.push("Clothing / dress references require Clothing / dress to be selected.");
    }
    const hasStylingReference = state.accessoryAssets.some((item) => item.status === "ready");
    const detailTargets = ["jewelry", "bags", "footwear", "makeup", "hair", "other"];
    if (hasStylingReference && !detailTargets.some((target) => selected.has(target))) {
      errors.push(
        "Styling detail references require Jewelry, Bags, Footwear, Makeup / lipstick, Hair, or Other styling to be selected.",
      );
    }
    return errors;
  }

  function buildChecks() {
    const checks = [];
    const member = selectedMember();
    const look = selectedValue("look_source");
    const audio = selectedValue("audio_mode");
    const duration = effectiveDuration();
    const longAckNeeded = duration !== null && duration > 10;
    const driverAudioMissing = audio === "driver" && state.driver && videoMetadata(state.driver).hasAudio === false;
    const memberRequired = look !== "exact_image" || audio === "cast_voice";

    checks.push({ready: Boolean(state.driver), error: false, label: state.driver ? "Driving video staged" : "Add a driving video"});
    const targetConfirmed = Boolean(state.driver) && $("#animate-target-confirmed").checked;
    checks.push({
      ready: targetConfirmed,
      error: false,
      label: targetConfirmed ? "Driver target person confirmed" : "Review and confirm the driver target person",
    });
    if (duration !== null && duration > state.maxDriverRangeSec) {
      checks.push({
        ready: false,
        error: true,
        label: `Select at most ${state.maxDriverRangeSec} seconds of driver footage`,
      });
    }
    checks.push({
      ready: Boolean(member) || !memberRequired,
      error: false,
      label: member
        ? "Target member selected"
        : memberRequired ? "Select one target member" : "Uploaded image supplies the character identity",
    });

    let lookReady = Boolean(member);
    let lookLabel = "Character look configured";
    if ((look === "auto_lora" || look === "styled_lora") && member?.has_lora === false) {
      lookReady = false;
      lookLabel = "Selected member needs a FLUX.2 LoRA";
    }
    if (look === "styled_lora" && lookReady) {
      const scopeErrors = completeLookScopeErrors();
      if (scopeErrors.length) {
        lookReady = false;
        lookLabel = scopeErrors[0];
      }
    }
    if (look === "exact_image") {
      lookReady = state.exactImage?.status === "ready";
      lookLabel = lookReady ? "Character image uploaded" : "Upload a character image";
    }
    checks.push({ready: lookReady, error: false, label: lookLabel});

    let audioReady = true;
    let audioLabel = "Audio choice is compatible";
    if (driverAudioMissing) {
      audioReady = false;
      audioLabel = "Driving video has no audio stream";
    } else if (audio === "cast_voice" && member?.has_voice === false) {
      audioReady = false;
      audioLabel = "Selected member needs a configured voice";
    }
    checks.push({ready: audioReady, error: !audioReady, label: audioLabel});

    if (longAckNeeded) {
      const ready = $("#animate-long-driver-ack").checked;
      checks.push({ready, error: false, label: ready ? "Long-job warning accepted" : "Confirm long-driver processing"});
    }
    if (state.pendingUploads || state.driverBusy) {
      checks.push({ready: false, error: false, label: "Wait for asset uploads to finish"});
    }
    if ($("#animate-lipsync-enabled").checked) {
      const backend = $("#animate-lipsync-backend").value;
      const backendReadiness = state.lipsyncReadiness[backend];
      if (backendReadiness?.ready === false) {
        checks.push({
          ready: false,
          error: true,
          label: backendReadiness.reason || `${backend} service is unavailable`,
        });
      }
    }
    if (state.backendReady === false) {
      checks.push({ready: false, error: true, label: state.backendReason || "Wan Animate service is unavailable"});
    }
    const rights = $("#animate-rights-cleared").checked;
    checks.push({ready: rights, error: false, label: rights ? "Usage rights confirmed" : "Confirm usage rights"});
    return checks;
  }

  function renderReadiness() {
    const duration = effectiveDuration();
    const longDriver = duration !== null && duration > 10;
    $("#animate-target-confirm").hidden = !state.driver;
    $("#animate-long-driver-confirm").hidden = !longDriver;
    if (!longDriver) $("#animate-long-driver-ack").checked = false;

    const checks = buildChecks();
    const container = $("#animate-readiness-list");
    container.replaceChildren(...checks.map((check) => {
      const row = document.createElement("div");
      row.className = `animate-check${check.ready ? " ready" : ""}${check.error ? " error" : ""}`;
      row.textContent = check.label;
      return row;
    }));
    const ready = checks.every((check) => check.ready) && !state.submitting;
    $("#animate-submit").disabled = !ready;

    const badge = $("#animate-readiness");
    badge.classList.remove("ready", "blocked");
    if (state.backendReady === false) {
      badge.classList.add("blocked");
      $("#animate-readiness-label").textContent = "Setup required";
    } else if (state.backendReady === true) {
      badge.classList.add("ready");
      $("#animate-readiness-label").textContent = "Service ready";
    } else {
      $("#animate-readiness-label").textContent = state.options ? "Readiness not reported" : "Checking readiness…";
    }
  }

  function renderSummary() {
    const mode = selectedValue("animate_mode");
    const look = selectedValue("look_source");
    const audio = selectedValue("audio_mode");
    const member = selectedMember();
    const cast = selectedCast();
    const metadata = videoMetadata(state.driver);
    const driverBits = state.driver ? [state.driver.name] : ["Not selected"];
    if (metadata.duration !== null) driverBits.push(formatDuration(metadata.duration));
    $("#summary-mode").textContent = mode === "replace" ? "Character replacement" : "Motion transfer";
    $("#summary-driver").textContent = driverBits.join(" · ");
    $("#summary-target").textContent = member
      ? `${member.name} · ${cast?.name || cast?.id}`
      : look === "exact_image" ? "From uploaded character image" : "Not selected";
    $("#summary-look").textContent = {
      auto_lora: "Auto-generate with LoRA",
      styled_lora: "Designed complete look + LoRA",
      exact_image: "Uploaded character image",
    }[look];
    $("#summary-audio").textContent = {
      driver: "Driving audio",
      cast_voice: "Cast voice · original words",
      none: "No audio",
    }[audio];
    $("#summary-lipsync").textContent = $("#animate-lipsync-enabled").checked
      ? $(`#animate-lipsync-backend option:checked`).textContent.split(" — ")[0]
      : "Off";
    const generation = $("#animate-generation-area").value;
    const exportMode = {
      generated: "generated size",
      scale_1080p: "1080p long edge",
      vertical_1080x1920: "1080 × 1920 fit",
    }[selectedValue("export_mode")] || "generated size";
    const fps = $("#animate-target-fps").value === "48" ? "48 FPS" : "generated FPS";
    $("#summary-output").textContent = `${generation} · ${exportMode} · ${fps}`;
  }

  function renderAll() {
    setModePanels();
    setLookPanel();
    renderFeatureAvailability();
    renderLipSync();
    renderMemberCapabilities();
    $("#animate-range-fields").hidden = $("#animate-timeline").value !== "selected_range";
    const retarget = $("#animate-retarget-pose").checked && selectedValue("animate_mode") === "animate";
    const fluxRetargetInput = $("#animate-flux-retarget");
    $("#animate-flux-retarget-row").hidden = !retarget;
    $("#animate-flux-retarget-hint").hidden = !retarget;
    $("#animate-flux-retarget-hint").textContent = state.wanFluxRetargetReason;
    fluxRetargetInput.disabled = state.wanFluxRetargetEnabled !== true;
    if (!retarget || state.wanFluxRetargetEnabled !== true) fluxRetargetInput.checked = false;
    $("#animate-use-server-file").disabled = state.driverBusy || !$("#animate-server-file").value;
    renderSummary();
    renderReadiness();
  }

  function validateForm() {
    setError("animate-driver-error", "");
    setError("animate-target-error", "");
    setError("animate-character-error", "");
    setError("animate-wardrobe-error", "");
    setError("animate-audio-error", "");
    setError("animate-advanced-error", "");
    setError("animate-submit-error", "");

    const errors = [];
    const look = selectedValue("look_source");
    const audio = selectedValue("audio_mode");
    const member = selectedMember();
    if (!state.driver) {
      const message = "Add and finish staging a driving video.";
      setError("animate-driver-error", message);
      errors.push(message);
    }
    if (state.driver && !$("#animate-target-confirmed").checked) {
      const message = "Review the driver preview and confirm the intended target person.";
      setError("animate-driver-error", message);
      errors.push(message);
    }
    if (!member && (look !== "exact_image" || audio === "cast_voice")) {
      const message = "Select exactly one target cast member.";
      setError("animate-target-error", message);
      errors.push(message);
    }
    if ((look === "auto_lora" || look === "styled_lora") && member?.has_lora === false) {
      errors.push("The selected member does not have a FLUX.2 LoRA.");
    }
    if (look === "styled_lora") {
      const scopeErrors = completeLookScopeErrors();
      if (scopeErrors.length) {
        setError("animate-wardrobe-error", scopeErrors[0]);
        errors.push(...scopeErrors);
      }
    }
    if (look === "styled_lora") {
      const listLimits = [
        ["wardrobe-jewelry", 12, "jewelry items"],
        ["wardrobe-bags", 8, "bags"],
        ["wardrobe-accessories", 12, "other accessories"],
      ];
      listLimits.forEach(([id, limit, label]) => {
        if (listFromCommaInput(id).length > limit) {
          const message = `Use no more than ${limit} comma-separated ${label}.`;
          setError("animate-wardrobe-error", message);
          errors.push(message);
        }
      });
    }
    if (look === "styled_lora"
        && state.garmentAssets.length + state.accessoryAssets.length > state.fluxEditMaxUserReferences) {
      const message = `Use no more than ${state.fluxEditMaxUserReferences} clothing/styling images combined.`;
      setError("animate-wardrobe-error", message);
      errors.push(message);
    }
    if (look === "exact_image" && state.exactImage?.status !== "ready") {
      const message = "Upload a valid character reference image.";
      setError("animate-character-error", message);
      errors.push(message);
    }
    if (audio === "driver" && state.driver && videoMetadata(state.driver).hasAudio === false) {
      const message = "This driving video is silent. Choose Cast voice or No audio.";
      setError("animate-audio-error", message);
      errors.push(message);
    }
    if (audio === "cast_voice" && member?.has_voice === false) {
      const message = "The selected member does not have a configured cast voice.";
      setError("animate-audio-error", message);
      errors.push(message);
    }
    if ($("#animate-lipsync-enabled").checked) {
      const backend = $("#animate-lipsync-backend").value;
      const backendReadiness = state.lipsyncReadiness[backend];
      if (backendReadiness?.ready === false) {
        const message = backendReadiness.reason || `${backend} service is unavailable.`;
        setError("animate-audio-error", message);
        errors.push(message);
      }
    }
    if ($("#animate-timeline").value === "selected_range") {
      const start = asNumber($("#animate-start-sec").value);
      const end = asNumber($("#animate-end-sec").value);
      const duration = state.driver ? videoMetadata(state.driver).duration : null;
      if (start === null || start < 0 || end === null || end <= start || (duration !== null && end > duration + .05)) {
        const message = "Choose a valid range inside the driving video; end must be after start.";
        setError("animate-advanced-error", message);
        errors.push(message);
      }
    }
    const selectedDuration = effectiveDuration();
    if (selectedDuration !== null && selectedDuration > state.maxDriverRangeSec) {
      const message = `Wan Animate jobs are limited to ${state.maxDriverRangeSec} seconds. Choose a shorter driver range.`;
      setError("animate-advanced-error", message);
      errors.push(message);
    }
    const steps = asNumber($("#animate-sampling-steps").value);
    if (steps === null || !Number.isInteger(steps) || steps < 10 || steps > 40) {
      const message = "Sampling steps must be a whole number from 10 to 40.";
      setError("animate-advanced-error", message);
      errors.push(message);
    }
    if (selectedValue("animate_mode") === "replace") {
      const maskValues = [
        ["animate-mask-iterations", 0, 10, "Mask iterations"],
        ["animate-mask-kernel", 1, 31, "Mask kernel"],
        ["animate-mask-width", 1, 8, "Mask width blend"],
        ["animate-mask-height", 1, 8, "Mask height blend"],
      ];
      for (const [id, min, max, label] of maskValues) {
        const value = asNumber($(`#${id}`).value);
        if (value === null || !Number.isInteger(value) || value < min || value > max) {
          const message = `${label} must be a whole number from ${min} to ${max}.`;
          setError("animate-advanced-error", message);
          errors.push(message);
          break;
        }
      }
      const kernel = asNumber($("#animate-mask-kernel").value);
      if (kernel !== null && kernel % 2 === 0) {
        const message = "Mask kernel must be an odd number.";
        setError("animate-advanced-error", message);
        errors.push(message);
      }
    }
    if (effectiveDuration() > 10 && !$("#animate-long-driver-ack").checked) errors.push("Accept the long-driver warning.");
    if (state.pendingUploads || state.driverBusy) errors.push("Wait for all uploads to finish.");
    if (state.backendReady === false) errors.push(state.backendReason || "Wan Animate is not ready.");
    if (!$("#animate-rights-cleared").checked) errors.push("Confirm that all source and character rights are cleared.");
    return [...new Set(errors)];
  }

  function listFromCommaInput(id) {
    return $(`#${id}`).value.split(",").map((item) => item.trim()).filter(Boolean);
  }

  function buildPayload() {
    const mode = selectedValue("animate_mode");
    const look = selectedValue("look_source");
    const audioMode = selectedValue("audio_mode");
    const cast = selectedCast();
    const member = selectedMember();
    const timeline = $("#animate-timeline").value;
    const driver = {
      asset_id: state.driver.asset_id,
      target_confirmed: true,
      timeline,
      subject_selection: $("#animate-subject").value,
    };
    if (timeline === "selected_range") {
      driver.start_sec = Number($("#animate-start-sec").value);
      driver.end_sec = Number($("#animate-end-sec").value);
    }

    const character = {
      look_source: look,
      cast_ref: member ? (cast?.id || null) : null,
      member_id: member?.id || null,
      consistency: "job",
    };
    if (look === "exact_image") character.exact_image_asset_id = state.exactImage.asset_id;
    if (look === "styled_lora") {
      character.wardrobe = {
        change_targets: selectedStyleTargets(),
        clothing_type: $("#wardrobe-clothing-type").value.trim(),
        primary_color: $("#wardrobe-primary-color").value.trim(),
        material_pattern: $("#wardrobe-material-pattern").value.trim(),
        jewelry: listFromCommaInput("wardrobe-jewelry"),
        bags: listFromCommaInput("wardrobe-bags"),
        footwear: $("#wardrobe-footwear").value.trim(),
        makeup: $("#wardrobe-makeup").value.trim(),
        hair: $("#wardrobe-hair").value.trim(),
        accessories: listFromCommaInput("wardrobe-accessories"),
        details: $("#wardrobe-details").value.trim(),
        negative_constraints: $("#wardrobe-negative").value.trim(),
        garment_asset_ids: state.fluxEditEnabled === true
          ? state.garmentAssets.filter((item) => item.status === "ready").map((item) => item.asset_id) : [],
        accessory_asset_ids: state.fluxEditEnabled === true
          ? state.accessoryAssets.filter((item) => item.status === "ready").map((item) => item.asset_id) : [],
      };
    }

    const advanced = {
      refert_num: Number($("#animate-refert-num").value),
      sampling_steps: Number($("#animate-sampling-steps").value),
    };
    if (mode === "animate") {
      advanced.retarget_pose = $("#animate-retarget-pose").checked;
      advanced.use_flux_retarget = $("#animate-flux-retarget").checked;
    } else {
      advanced.mask_iterations = Number($("#animate-mask-iterations").value);
      advanced.mask_kernel = Number($("#animate-mask-kernel").value);
      advanced.mask_w_len = Number($("#animate-mask-width").value);
      advanced.mask_h_len = Number($("#animate-mask-height").value);
    }

    return {
      workflow_kind: "wan_animate_direct",
      rights_cleared: true,
      animate: {
        schema_version: 1,
        mode,
        driver,
        character,
        audio: {
          mode: audioMode,
          ...(audioMode === "cast_voice" ? {voice_member_id: member.id} : {}),
          script_policy: "verbatim",
          timing: "match_driver",
        },
        lipsync: {
          enabled: $("#animate-lipsync-enabled").checked,
          backend: $("#animate-lipsync-backend").value,
        },
        output: {
          generation_area: $("#animate-generation-area").value,
          export: selectedValue("export_mode"),
          preserve_aspect: true,
          target_fps: $("#animate-target-fps").value === "48" ? 48 : "generated",
        },
        advanced,
      },
    };
  }

  async function submitJob(event) {
    event.preventDefault();
    const errors = validateForm();
    if (errors.length) {
      setError("animate-submit-error", errors.join("\n"));
      renderAll();
      $("#animate-submit-error").scrollIntoView({behavior: "smooth", block: "center"});
      return;
    }

    state.submitting = true;
    const button = $("#animate-submit");
    button.textContent = "Creating job…";
    renderAll();
    try {
      const payload = await requestJson(URLS.create, {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(buildPayload()),
      });
      const jobId = payload.job_id || payload.job?.job_id;
      if (!jobId) throw new Error("The job was created, but the server did not return a job ID.");
      window.location.assign(`/jobs/${encodeURIComponent(jobId)}`);
    } catch (error) {
      setError("animate-submit-error", error.message);
      state.submitting = false;
      button.textContent = "Create Animate job";
      renderAll();
    }
  }

  function bindEvents() {
    const driverTabs = $$('[data-driver-tab]');
    driverTabs.forEach((button, index) => {
      button.addEventListener("click", () => chooseDriverTab(button.dataset.driverTab));
      button.addEventListener("keydown", (event) => {
        if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
        event.preventDefault();
        let targetIndex = index;
        if (event.key === "ArrowLeft") targetIndex = (index - 1 + driverTabs.length) % driverTabs.length;
        if (event.key === "ArrowRight") targetIndex = (index + 1) % driverTabs.length;
        if (event.key === "Home") targetIndex = 0;
        if (event.key === "End") targetIndex = driverTabs.length - 1;
        driverTabs[targetIndex].focus();
        chooseDriverTab(driverTabs[targetIndex].dataset.driverTab);
      });
    });
    $("#animate-import-url").addEventListener("click", importDriverUrl);
    $("#animate-driver-url").addEventListener("keydown", (event) => {
      if (event.key === "Enter") {
        event.preventDefault();
        importDriverUrl();
      }
    });
    $("#animate-driver-file").addEventListener("change", (event) => uploadDriver(event.target.files?.[0]));
    const dropzone = $("#animate-driver-dropzone");
    ["dragenter", "dragover"].forEach((type) => dropzone.addEventListener(type, (event) => {
      event.preventDefault();
      dropzone.classList.add("dragover");
    }));
    ["dragleave", "drop"].forEach((type) => dropzone.addEventListener(type, (event) => {
      event.preventDefault();
      dropzone.classList.remove("dragover");
    }));
    dropzone.addEventListener("drop", (event) => uploadDriver(event.dataTransfer?.files?.[0]));
    $("#animate-remove-driver").addEventListener("click", removeDriver);
    $("#animate-refresh-server-files").addEventListener("click", loadServerFiles);
    $("#animate-server-file").addEventListener("change", renderAll);
    $("#animate-use-server-file").addEventListener("click", importServerFile);

    $("#animate-cast").addEventListener("change", () => populateMembers());
    $("#animate-member").addEventListener("change", () => {
      setError("animate-target-error", "");
      renderAll();
    });
    $("#wardrobe-garment-files").addEventListener("change", (event) => {
      addReferenceImages(event.target.files, "garment");
      event.target.value = "";
    });
    $("#wardrobe-accessory-files").addEventListener("change", (event) => {
      addReferenceImages(event.target.files, "accessory");
      event.target.value = "";
    });
    $("#animate-character-file").addEventListener("change", (event) => uploadExactImage(event.target.files?.[0]));
    root.addEventListener("click", (event) => {
      const button = event.target.closest("[data-remove-image]");
      if (button) removeReferenceImage(button.dataset.removeImage, button.dataset.imageKind);
    });

    $$('input[name="audio_mode"]').forEach((input) => input.addEventListener("change", applyAudioDefaults));
    $("#animate-lipsync-enabled").addEventListener("change", renderAll);
    $("#animate-retarget-pose").addEventListener("change", renderAll);
    $("#animate-timeline").addEventListener("change", renderAll);
    $("#animate-subject").addEventListener("change", () => {
      $("#animate-target-confirmed").checked = false;
      renderAll();
    });
    $("#animate-driver-preview").addEventListener("loadedmetadata", (event) => {
      if (!state.driver) return;
      const metadata = state.driver.metadata;
      if (!asNumber(metadata.duration_sec) && Number.isFinite(event.target.duration)) metadata.duration_sec = event.target.duration;
      if (!asNumber(metadata.width)) metadata.width = event.target.videoWidth;
      if (!asNumber(metadata.height)) metadata.height = event.target.videoHeight;
      renderDriverMetadata();
      renderAll();
    });

    root.addEventListener("input", (event) => {
      if (event.target.matches("input, textarea, select")) {
        if (event.target.id.startsWith("wardrobe-")) setError("animate-wardrobe-error", "");
        if (event.target.id.startsWith("animate-mask-")
            || ["animate-start-sec", "animate-end-sec", "animate-sampling-steps"].includes(event.target.id)) {
          setError("animate-advanced-error", "");
        }
        renderAll();
      }
    });
    root.addEventListener("change", (event) => {
      if (event.target.name === "style_change_target") {
        setError("animate-wardrobe-error", "");
      }
      if (event.target.matches("input, textarea, select")) renderAll();
    });
    $("#animate-job-form").addEventListener("submit", submitJob);
    window.addEventListener("beforeunload", () => {
      if (state.driverPreviewObjectUrl) URL.revokeObjectURL(state.driverPreviewObjectUrl);
      [...state.garmentAssets, ...state.accessoryAssets, state.exactImage].filter(Boolean)
        .forEach((item) => item.preview_url && URL.revokeObjectURL(item.preview_url));
    });
  }

  bindEvents();
  renderAll();
  loadOptions();
  loadServerFiles();
})();
