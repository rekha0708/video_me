"""PyTorch 2.6+ changed torch.load's weights_only default from False to True,
which breaks mmengine/MuseTalk checkpoint loading (they load full pickled
objects, not just tensor weights). MuseTalk's inference.py runs as a separate
subprocess, so this can't be a plain in-process monkey-patch from
musetalk_server.py — it has to apply inside that subprocess's own interpreter.

`sitecustomize` is a standard Python convention: any module named
sitecustomize.py in a directory on sys.path gets auto-imported at interpreter
startup. musetalk_server.py puts this directory first on the subprocess's
PYTHONPATH, so this patch applies before mmengine/musetalk ever call
torch.load — without editing any installed package or vendored file.

These checkpoints are all fetched by our own setup_gpu.sh from named,
trusted HF/PyTorch-CDN sources (see setup_musetalk() and the model
download list) — not arbitrary user-supplied files — so relaxing
weights_only here is scoped to a known-safe case, not a general bypass.
"""
import torch

_orig_load = torch.load


def _patched_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_load(*args, **kwargs)


torch.load = _patched_load
