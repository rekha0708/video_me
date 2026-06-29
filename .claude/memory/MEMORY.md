# Memory Index — video_me project

- [Project: video_me pipeline](project_video_me.md) — Stack, services, key paths, GPU environment
- [Project: LoRA training](project_lora_training.md) — musubi-tuner config, VRAM breakdown, lessons learned, current training status
- [Project: Wan 2.2](project_wan22.md) — Installation quirks, nested path gotcha, missing deps
- [Project: Next steps](project_next_steps.md) — Post-training checklist; source video path confirmed
- [Feedback: Training optimizer/precision](feedback_training.md) — Use adamw (not adamw8bit); keep fp8_base on H200
- [Feedback: Render adapter](feedback_render_adapter.md) — Use musubi_flux not comfyui_flux; ComfyUI has no Mistral 3 loader
