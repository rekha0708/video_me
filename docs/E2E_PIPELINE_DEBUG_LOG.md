# End-to-End Pipeline Debug Log

Running log of every bug found and fixed while getting a real dashboard job
(`20260703-203517-cbe`) through the full pipeline for the first time. Kept
updated as we go — newest entries at the bottom of each session's block.

---

## Session: 2026-07-03

### 1. Dashboard stuck after plan approval

**Symptom:** operator approves the storyboard in the dashboard; job status
never leaves `pending_plan_approval`.

**Root cause:** `DashboardRepository.get_pending_approval(job_id)` filters
`WHERE status='pending'` in SQL. The instant an approval is resolved
(approved/rejected), it vanishes from that query. Three separate poll loops
called this to check for a decision on an approval they'd already created:
- `adapters/approval/dashboard_approval_adapter.py` (plan approval)
- `adapters/approval/dashboard_image_approval_adapter.py` (image approval)
- `services/dashboard_worker.py` `_poll_for_approval` (transcript review gate)

Each one just saw `None` forever and hung until a 4h timeout.

**Fix:** added `DashboardRepository.get_approval(approval_id)` (fetch by ID
regardless of status) and switched all three poll loops to it.
**Status:** ✅ Fixed and verified — plan approval now resolves in ~5s.

### 2. `.env` render adapter drifted to the broken ComfyUI Flux path

**Symptom:** render stage fails with `HTTPStatusError 400` from
`http://localhost:8188/prompt`.

**Root cause:** `.env` had `VIDEO_ME_RENDER_ADAPTER=comfyui_flux`, but
ComfyUI can't run Flux 2.0 at all — no Mistral 3 text encoder node, and the
checked-in workflow (`assets/comfyui_workflows/flux_lora_txt2img.json`) is a
leftover Flux-1.x template (DualCLIPLoader + T5/CLIP) with only the UNET
filename swapped to a Flux 2.0 checkpoint. The referenced CLIP/T5 files don't
even exist in `models/clip/`.

**Fix:** `.env`: `VIDEO_ME_RENDER_ADAPTER=musubi_flux` (the documented working
default).
**Status:** ✅ Fixed.

### 3. `musubi_tuner` not installed anywhere

**Symptom:** render stage fails with `ModuleNotFoundError: No module named
'musubi_tuner'`.

**Root cause:** `adapters/render_character/musubi_flux_adapter.py` invoked
the render script via `sys.executable` — i.e. whichever Python runs the
dashboard worker (`/workspace/video_me/.venv`, a real isolated venv with
`include-system-site-packages=false`). `musubi_tuner` was never installed
there, and `setup_gpu.sh` used to install it into system `pip3` — which
silently vanishes on a pod restart (same class of loss as Ollama's binary).

**Fix:**
- Created `/workspace/.venv_musubi` (`python3 -m venv --system-site-packages`,
  inherits torch 2.8.0+cu128), installed `musubi-tuner` + `flash-attn` there.
- `musubi_flux_adapter.py`: added `_MUSUBI_PYTHON` constant pointing at that
  venv, replaced `sys.executable`.
- `setup_gpu.sh`'s `setup_musubi_tuner()`: now creates/installs into this
  dedicated venv instead of system pip3.
- `start_services.sh`: added a preflight check that verifies
  `.venv_musubi` can still import `musubi_tuner`, self-heals (reinstall) if not.
**Status:** ✅ Fixed.

### 4. `--text_encoder` pointed at a directory, not the shard file

**Symptom:** `IsADirectoryError: [Errno 21] Is a directory:
'/workspace/FLUX2-text-encoder'`.

**Root cause:** musubi-tuner's `load_split_weights()` expects the path to the
*first* shard file (`model-00001-of-00010.safetensors`) and derives sibling
filenames by pattern-matching — it does not accept the parent directory.

**Fix:** `musubi_flux_adapter.py`: `_TEXT_ENCODER` now points directly at
`/workspace/FLUX2-text-encoder/text_encoder/model-00001-of-00010.safetensors`.
The tokenizer is pulled separately from HF Hub
(`mistralai/Mistral-Small-3.1-24B-Instruct-2503`), not from the local
`tokenizer/` subdir, so no path needed for that.
**Status:** ✅ Fixed.

### 5. `flash_attn_func` was `None`

**Symptom:** `TypeError: 'NoneType' object is not callable` deep in
`flux2_models.py`'s attention forward pass.

**Root cause:** `flash-attn` was never installed into the new
`.venv_musubi` (only the base `pip install -e musubi-tuner` had been run).

**Fix:** `.venv_musubi/bin/pip install flash-attn --no-build-isolation` —
resolved instantly from a cached prebuilt wheel, no compilation needed.
**Status:** ✅ Fixed.

### 6. `--save_path` treated as a file, but musubi-tuner treats it as a directory

**Symptom:** `IsADirectoryError: [Errno 21] Is a directory:
'.../render_00.png'` — one stage later, in image critique, because
`render_00.png` was actually created as a *directory*.

**Root cause:** `flux_2_generate_image.py` does
`os.makedirs(save_path, exist_ok=True)` and invents its own filename inside
it (`save_images_grid`: `"{time_flag}_{seed}_000.png"`) — it does not accept
a literal target file path.

**Fix:** `musubi_flux_adapter.py._generate()`: pass a scratch directory as
`--save_path`, then `shutil.move()` the single generated PNG to the expected
flat `out_path`, then remove the scratch dir.
**Status:** ✅ Fixed and verified — 3 real candidate images generated for shot s01.

### 7. `ImageCritiqueResult` had no `candidate_uris` field

**Symptom:** `ValueError: "ImageCritiqueResult" object has no field
"candidate_uris"` — right after a successful VLM critique.

**Root cause:** `core/workflow.py` tried to monkeypatch-assign
`critique.candidate_uris = render_result.images` after construction, but
`ImageCritiqueResult` (Pydantic model) never declared that field. Two more
bugs downstream depended on the same missing data:
`adapters/approval/dashboard_image_approval_adapter.py` read
`s.uri`/`s.character_likeness`/`s.scene_match`/`s.composition`/`s.overall`
off `ImageCandidateScore`, none of which exist (real fields:
`candidate_index`, `scores: dict`, `reasoning`).

**Fix:**
- `core/models/capabilities.py`: added a real `candidate_uris: list[str]`
  field to `ImageCritiqueResult`.
- `adapters/critique/image_critique_adapter.py`: populate it at construction
  (both the JSON-repair-failure fallback and the normal return path).
- `core/workflow.py`: removed the broken monkeypatch line.
- `dashboard_image_approval_adapter.py`: fixed the payload builder and the
  operator-override path to use the real fields.
**Status:** ✅ Fixed and verified — all 14 shots cleared render + critique,
job reached `pending_image_approval` with a well-formed payload (checked the
DB directly: all 14 shots have 3 `candidate_uris` + a `vlm_winner_index`).

### 8. Rendered images don't display in the image approval UI

**Symptom:** dashboard shows the image approval card, but candidate images
don't render.

**Root cause:** `approval_images.html` did `<img src="{{ uri }}">` with the
raw local filesystem path (e.g. `.local/jobs/.../renders/max/render_00.png`)
— not a URL the browser can fetch. There was no route in `dashboard_api.py`
to serve arbitrary render files at all (only `/static` for CSS/JS). The CLI's
`image_approval_adapter.py` solves this with its own mini HTTP server and a
`/img/<base64-encoded-path>` route; the dashboard API had no equivalent.

**Fix:** `services/dashboard_api.py`:
- Registered a Jinja filter `b64path` (base64url-encodes a path).
- Added `GET /img/{path_b64}`: decodes the path, validates it resolves
  *inside* `settings.data_dir` (path-traversal guard), returns a
  `FileResponse`.
- `approval_images.html`: `<img src="/img/{{ uri | b64path }}">`.

**Status:** ✅ Fixed and verified — `curl`'d the route directly, got back a
real 1024×1024 PNG; job detail page renders with no template errors.

### 9. Orphaned zombie process squatting on port 8765 across every restart

**Symptom:** after fixing #8 and restarting, the fix appeared to have *no
effect* — same `TemplateAssertionError: No filter named 'b64path'` from a
"freshly restarted" server.

**Root cause:** a `multiprocessing.spawn` child process (PID from ~20:38,
over an hour earlier) was still bound to port 8765 and serving requests with
long-stale code. It was almost certainly an orphaned child leaked by an
earlier `uvicorn --reload` reloader restart. `restart_dashboard.sh`'s
`stop_matching()` kills by matching `pgrep -f "uvicorn services.dashboard_api"`
against the process cmdline — but this orphan's cmdline was
`python -c "from multiprocessing.spawn import spawn_main; ..."`, which
doesn't contain that string, so it was invisible to every restart attempt
and kept squatting on the port indefinitely. New start attempts were
actually failing with `[Errno 98] Address already in use` in the log the
whole time — `curl` kept getting 200s from the *zombie*, not the new process.

**Fix:** `scripts/restart_dashboard.sh`: added `stop_port()` — finds PIDs
actually bound to `$DASHBOARD_PORT` via `ss -ltnp` and force-kills them,
as a second layer alongside the existing pattern-based kill.

**Lesson:** when a restart doesn't seem to take effect, check
`ss -ltnp | grep <port>` for the actual PID serving the port before assuming
the code fix is wrong — don't trust `curl` 200s alone as proof the *new*
process is answering.

**Status:** ✅ Fixed and verified.
