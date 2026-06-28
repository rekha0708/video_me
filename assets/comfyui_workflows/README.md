# ComfyUI Workflow Templates

Place exported ComfyUI workflow JSON files here. The adapters load these templates
and substitute placeholder node titles at runtime.

## flux_lora_txt2img.json — render_character (ComfyUIFluxAdapter)

Export your Flux 2.0 Dev txt2img + LoRA workflow from ComfyUI (Settings → Save as API format).
The adapter looks for nodes whose `_meta.title` matches these sentinels:

| Sentinel title  | Node type                  | Field substituted      |
|-----------------|----------------------------|------------------------|
| `__PROMPT__`    | CLIPTextEncodeFlux         | `text`                 |
| `__LORA_NAME__` | LoraLoader                 | `lora_name`, `strength_model`, `strength_clip` |
| `__WIDTH__`     | EmptySD3LatentImage        | `width`                |
| `__HEIGHT__`    | EmptySD3LatentImage        | `height`               |
| `__STEPS__`     | BasicScheduler / KSampler  | `steps`                |
| `__SEED__`      | RandomNoise                | `noise_seed`           |

**How to set a sentinel title in ComfyUI:** right-click the node → "Title" → type the sentinel.

If this file is absent the adapter falls back to a minimal hard-coded workflow (no LoRA).

## ltx_i2v.json — generate_video (LtxAdapter)

Export your LTX-2.3 22B distilled image-to-video workflow from ComfyUI in API format.
Sentinel titles:

| Sentinel title | Node type              | Field substituted |
|----------------|------------------------|-------------------|
| `__IMAGE__`    | LoadImage              | `image`           |
| `__AUDIO__`    | LoadAudio              | `audio`           |
| `__PROMPT__`   | CLIPTextEncode         | `text`            |
| `__FRAMES__`   | LTXVConditioning       | `length`          |
| `__STEPS__`    | LTXVScheduler          | `steps`           |
| `__SEED__`     | RandomNoise            | `noise_seed`      |

If this file is absent the adapter falls back to a minimal hard-coded LTX workflow.
