<!-- CURATED (hand-written) — update when limitations change. Not touched by generate_code_map.py -->

# Known limitations & weak points (as of 2026-07-08)

Read this before debugging a stage or planning a feature that touches it.

## Per-stage weak points

| Stage / area | Limitation |
|---|---|
| render_character (default `musubi_flux`) | Runs musubi-tuner as a **subprocess** — no health endpoint, failures surface as non-zero exit + stderr parsing. One 64 GB model load per `run_many()` batch; a single-shot rerun pays the full load cost again. |
| render_character (`comfyui_flux` fallback) | ComfyUI **cannot load Flux 2.0 locally** — no Mistral 3 encoder node; `Flux2*` nodes call the paid BFL cloud API. Fallback in name only unless BFL API is acceptable. |
| render_character candidates | Default 1 candidate/shot (Flux candidates near-identical; operator decision 2026-07-07). With 1 candidate the VLM critique is **skipped** (`origin="single"`) — the human image gate is the only quality check. |
| generate_video (`ltx`) | LTX-2.3 22B via ComfyUI, 8-step distilled. Native lip-sync means **no separate lip_sync correction pass** — bad sync requires regenerating the whole clip. Completion marker is `clip.mp4`. |
| generate_video (`wan` fallback) | Needs `core/gpu_sequencer.py` deferred load/unload dance (`/load`/`/unload`, 409 = busy) to fit VRAM; ~21 min/shot vs ~1 min with LTX. Completion marker `synced.mp4`. |
| synthesize_voice | Voice WAVs are **gTTS bootstrap** — pipeline runs, but voices are not brand-accurate child voices. Fish S2 server (port 8025) must be started manually. |
| LLM stages (qwen3.6:35b) | Thinking mode must stay disabled (`extra_body={"think": False}`, no `response_format`); JSON output is repaired via `json_repair` — malformed JSON can still slip through as semantically wrong-but-valid. |
| plan critique loop | Max 3 re-plan iterations, all 5 dimensions ≥ 0.75; a systematically harsh/lenient LLM judge silently shifts quality. 2nd human rejection = job FAILED (no appeal path). |
| analyze_visuals | Best-effort: empty for story jobs; frame sampling from source video only. |
| Track B | `loras/kids_duo_max.safetensors` **missing**; `kids_duo_zoe.safetensors` is a TEST-ONLY placeholder. SD 1.5 LoRAs are incompatible with Flux 2.0 — retrain required. `VIDEO_ME_RENDER_ALLOW_PLACEHOLDER_LORA=true` bypass exists for smoke tests only. |

## Coupling & fragility

- **Shared approval port 8765**: storyboard gate and image gate reuse the same port sequentially. A crashed gate can leave the port bound. Dashboard-integrated approvals use port 8080 instead.
- **ComfyUI (8188) double duty**: LTX video *and* the (non-functional locally) Flux fallback. LTX is the only real reason it must run.
- **Pod restarts wipe Ollama**: base Linux binary is deleted on RunPod restart; `scripts/start_services.sh` reinstalls it. Models persist at `/workspace/ollama/`.
- **VRAM budget is tight**: ~114 GB peak of 143 GB (qwen3.6 30 + LTX 44 + Flux 20 + Fish 20). Adding any resident model risks OOM; the GPU sequencer only coordinates the Wan path.
- **Per-service venvs**: heavy ML deps live in separate venvs (`/workspace/.venv_*`); installing ML packages into the project `.venv` breaks the fast-CI invariant.
- **Story-kind jobs** are restricted to `transcribe`/`all` phases; later phases need upstream artifacts (use Advance on an existing job).
- **Stage-ordering contract**: adapters raise `FileNotFoundError("upstream_stage must run before this_stage")` — resume logic depends on completion-marker files, so manually deleting work-dir files can confuse `--resume-job`.

## Docs drift risks (why docs/code_map/ exists)

- CLAUDE.md previously claimed capability ABCs live in `core/capabilities/<stage>.py`; they are all in `core/capabilities/base.py`.
- Newer modules (`adapters/transcript_refine/`, `adapters/render_overlays/`, `adapters/approval/dashboard_approval_adapter.py`, `core/registry.py`, `core/router.py`, `services/dashboard_worker.py`, `services/dashboard_repository.py`, `services/chat_service.py`) were missing from the CLAUDE.md key-file map.
- The generated pages in this directory are enforced fresh by `tests/test_code_map.py`; this file is **not** — review it when a limitation above is fixed.
