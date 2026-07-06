"""render_overlays: deterministic chart/diagram panels for shot overlays.

Diffusion models cannot render legible charts — overlays are drawn with
matplotlib (CPU, no network) and composited over the video by ffmpeg at
assemble time. Best-effort everywhere: failures skip the panel, never the job.
"""
