"""Runs at interpreter start in the subprocesses wsitrain launches.

CellViT is invoked as a separate process and never imports wsitrain, so its
own progress bars miss the shared tqdm hardening. `site` imports
``sitecustomize`` before any user code, which makes a directory on PYTHONPATH
the one hook that lands early enough to matter.
"""
try:
    from wsitrain import _fit_torchinfo_to_terminal, _harden_tqdm_against_resize
except Exception:  # not importable in the child: leave tqdm at its defaults
    pass
else:
    _harden_tqdm_against_resize()
    _fit_torchinfo_to_terminal()
