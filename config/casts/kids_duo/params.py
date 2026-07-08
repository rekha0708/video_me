"""Render + voice asset params for the kids_duo cast (Max + Zoe).

Source of truth for this cast's trained assets and render tuning. Character
design/personality lives in config/casts/kids_duo.yaml.

- lora_file:      safetensor filename under settings.lora_dir (VIDEO_ME_LORA_DIR)
- lora_weight:    LoRA multiplier (None → adapter default)
- steps:          Flux sampling steps (None → adapter default)
- guidance_scale: Flux guidance (None → adapter default)
- trigger:        optional token(s) prepended to the render prompt
- style_suffix:   appended to the render prompt ("" → adapter's cartoon default)
- voice_file:     voice_profile_ref-form value under settings.voice_dir
                  (resolves to <voice_dir>/<ref>.wav), overrides the YAML

Track B: kids_duo_max.safetensors is currently MISSING and
kids_duo_zoe.safetensors is a TEST-ONLY placeholder — train both Flux 2.0 LoRAs.
"""

_CARTOON_STYLE = (
    "children's animation style, cartoon, vibrant colors, "
    "high quality, clean lines, expressive character"
)

MEMBERS = {
    "max": {
        "lora_file": "kids_duo_max.safetensors",
        "lora_weight": 0.9,
        "steps": 20,
        "guidance_scale": 3.5,
        "trigger": "",
        "style_suffix": _CARTOON_STYLE,
        "voice_file": "voices/kids_duo/max",
    },
    "zoe": {
        "lora_file": "kids_duo_zoe.safetensors",
        "lora_weight": 0.9,
        "steps": 20,
        "guidance_scale": 3.5,
        "trigger": "",
        "style_suffix": _CARTOON_STYLE,
        "voice_file": "voices/kids_duo/zoe",
    },
}

# Pair LoRA for two-character shots. Keyed by frozenset of member IDs.
# Empty lora_file until a pair LoRA is trained — the workflow falls back to
# speaker-only LoRA + prompt-append for the second character.
PAIRS = {
    frozenset({"max", "zoe"}): {
        "lora_file": "",
        "lora_weight": 0.9,
        "steps": 20,
        "guidance_scale": 3.5,
        "trigger": "",
    },
}
