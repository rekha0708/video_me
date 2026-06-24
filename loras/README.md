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

Current local status as of 2026-06-24: real trained weights exist at both expected
paths. They were trained locally with sd-scripts for 1000 steps each, rank 32,
against the SD 1.5 `v1-5-pruned-emaonly.safetensors` base model. Keep these files
on the GPU workspace or move them to the future asset store; they will not be
pushed to git.

Strict readiness should pass the LoRA checks and then continue to the remaining
Track B voice checks:

```bash
python -m scripts.check_runtime_readiness
```

Run this from the repo root after placing weights:

```bash
python -m scripts.check_track_b
```
