# Track B LoRA Weights

`kids_duo` is the final cast for the current pipeline. Place trained character
LoRA files here using the exact stems derived from `config/casts/kids_duo.yaml`:

```text
loras/
  kids_duo_max.safetensors
  kids_duo_zoe.safetensors
```

The adapter also accepts `.pt` or `.ckpt`, but `.safetensors` is preferred.
These binary files are intentionally ignored by git.

For development only, tiny placeholder `.safetensors` files may exist locally to
pass the file-gate preflight. They are not trained LoRAs and will not work in
AUTOMATIC1111. Replace them with real weights before any real render run.

Temporary smoke tests may opt into placeholder mode:

```bash
export VIDEO_ME_RENDER_ALLOW_PLACEHOLDER_LORA=true
bash scripts/setup_gpu.sh --code-test --skip-services
```

In that mode the render adapter omits the fake LoRA tag from the prompt. Strict
readiness still fails placeholders:

```bash
python -m scripts.check_runtime_readiness
```

Run this from the repo root after placing weights:

```bash
python -m scripts.check_track_b
```
