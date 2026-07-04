# Memory Index — video_me project

- [Project: video_me pipeline](project_video_me.md) — Stack, services, key paths, GPU environment
- [Project: LoRA training](project_lora_training.md) — musubi-tuner config, VRAM breakdown, lessons learned, current training status
- [Project: Wan 2.2](project_wan22.md) — Video gen works but MuseTalk lip-sync NOT viable; wan+musetalk resident VRAM caused OOM in render_character (2026-07-04)
- [Project: Next steps](project_next_steps.md) — Track B READY; job 20260704-020418-33o failed on VRAM OOM, decision pending; dashboard is on port 8765 not 8080
- [Feedback: Training optimizer/precision](feedback_training.md) — Use adamw (not adamw8bit); keep fp8_base on H200
- [Feedback: Render adapter](feedback_render_adapter.md) — Use musubi_flux not comfyui_flux; ComfyUI has no Mistral 3 loader
- [Feedback: musubi render perf](feedback_musubi_render_perf.md) — ~4min/candidate is disk I/O + fp8 quant, not a hang; no model residency between subprocess calls
- [Feedback: Service startup gaps](feedback_service_startup_gaps.md) — start_services.sh failures are usually 1-2 missing pip pkgs, not a full wipe; check logs first
- [Feedback: Dashboard worker](feedback_dashboard_worker.md) — restart after every pull (scripts/restart_dashboard.sh); approval-polling bug class; /retry endpoint semantics
