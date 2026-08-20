"""kurtorank — pan-tissue ensemble subtype annotation for gene-limited spatial transcriptomics."""
__version__ = "3.1.0"


def _harden_tqdm_against_resize() -> None:
    """Make every tqdm bar survive terminal / tmux resizes.

    1. ``dynamic_ncols=True`` becomes the default for every bar, so tqdm
       re-queries the terminal width on each refresh instead of caching the
       width it saw at construction time.
    2. A ``SIGWINCH`` handler redraws every live bar the moment the terminal
       is resized, rather than waiting for the next ``update()``.

    Sentinels are deliberately NOT package-prefixed: these packages share one
    env and land in one process, so a per-package sentinel would let each of
    them wrap ``tqdm.__init__`` and chain another SIGWINCH handler. Keep this
    block identical across wsinsight, sptxinsight, hplot, kurtorank, wsitrain.
    """
    try:
        from tqdm import std as _tqdm_std
    except Exception:
        return

    if not getattr(_tqdm_std.tqdm, "_tqdm_resize_hardened", False):
        _orig_init = _tqdm_std.tqdm.__init__

        def _init(self, *args, **kwargs):  # noqa: ANN001
            kwargs.setdefault("dynamic_ncols", True)
            _orig_init(self, *args, **kwargs)

        _tqdm_std.tqdm.__init__ = _init
        _tqdm_std.tqdm._tqdm_resize_hardened = True

    try:
        import signal

        if not hasattr(signal, "SIGWINCH"):
            return  # not POSIX (e.g. Windows); nothing to do
        if getattr(_tqdm_std.tqdm, "_tqdm_winch_installed", False):
            return

        _prev_handler = signal.getsignal(signal.SIGWINCH)

        def _on_winch(signum, frame):  # noqa: ANN001
            try:
                for inst in list(getattr(_tqdm_std.tqdm, "_instances", [])):
                    inst.clear(nolock=True)
                    inst.refresh(nolock=True)
            except Exception:
                pass
            if callable(_prev_handler):
                _prev_handler(signum, frame)

        signal.signal(signal.SIGWINCH, _on_winch)
        _tqdm_std.tqdm._tqdm_winch_installed = True
    except (ValueError, OSError):
        # signal.signal raises ValueError off the main thread; ignore.
        pass


_harden_tqdm_against_resize()


def rerank_markers(*args, **kwargs):
    """Lazy re-export of :func:`kurtorank.rank.main.rerank_markers`.

    Imported lazily so ``import kurtorank`` does not require the full
    Census stack when only the annotate pipeline is used.
    """
    from kurtorank.rank.main import rerank_markers as _impl
    return _impl(*args, **kwargs)


def build_panel(*args, **kwargs):
    """Lazy re-export of :func:`kurtorank.seed.main.build_panel`."""
    from kurtorank.seed.main import build_panel as _impl
    return _impl(*args, **kwargs)


__all__ = ["__version__", "rerank_markers", "build_panel"]
