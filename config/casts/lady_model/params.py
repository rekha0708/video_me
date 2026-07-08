"""Render + voice asset params for the lady_model cast (Meera).

Source of truth for this cast's trained assets and render tuning. Character
design/personality lives in config/casts/lady_model.yaml.

- lora_file:      safetensor filename under settings.lora_dir (VIDEO_ME_LORA_DIR)
- lora_weight:    LoRA multiplier (None → adapter default)
- steps:          Flux sampling steps (None → adapter default)
- guidance_scale: Flux guidance (None → adapter default)
- trigger:        optional token(s) prepended to the render prompt
- style_suffix:   appended to the render prompt — Meera's LoRA was trained on
                  real photos, so this overrides the adapters' cartoon-style
                  default (which is meant for the original kids_duo cast).
- voice_file:     voice_profile_ref-form value under settings.voice_dir
                  (resolves to <voice_dir>/<ref>.wav), overrides the YAML
"""

MEMBERS = {
    "Meera": {
        "lora_file": "lady_model_meera.safetensors",
        "lora_weight": 0.9,
        "steps": 20,
        "guidance_scale": 3.5,
        "trigger": "",
        "style_suffix": (
            "photorealistic, cinematic lighting, natural skin texture, "
            "sharp focus, high detail, realistic photography"
        ),
        "voice_file": "voices/lady_model/meera",
    },
}
