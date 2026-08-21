"""wsinsight-train — headless CLI to train WSInsight CellViT heads."""

__version__ = "0.1.0"


def _harden_tqdm_against_resize() -> None:
    """Make every tqdm bar survive terminal / tmux resizes.

    1. ``dynamic_ncols=True`` and ``ascii=" ="`` become the defaults for every
       bar, so tqdm re-queries the terminal width on each refresh instead of
       caching the width it saw at construction time, and third-party bars
       (cellpose, stardist, torch) draw in the same style as ours instead of
       the default unicode blocks.
    2. A ``SIGWINCH`` handler redraws every live bar the moment the terminal
       is resized, rather than waiting for the next ``update()`` -- which on
       the segment stage can be minutes away. A resize that lands while the
       main thread sits in a long C call (CUDA, tifffile) is still only picked
       up when that call returns: Python runs handlers between bytecodes.

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
            kwargs.setdefault("ascii", " =")
            _orig_init(self, *args, **kwargs)

        _tqdm_std.tqdm.__init__ = _init
        _tqdm_std.tqdm._tqdm_resize_hardened = True

    try:
        import os
        import signal

        if not hasattr(signal, "SIGWINCH"):
            return  # not POSIX (e.g. Windows); nothing to do
        if getattr(_tqdm_std.tqdm, "_tqdm_winch_installed", False):
            return

        _prev_handler = signal.getsignal(signal.SIGWINCH)

        def _on_winch(signum, frame):  # noqa: ANN001
            # tqdm falls back to COLUMNS/LINES when the ioctl fails (redirected
            # fp); a stale pair exported by the shell would pin the old width.
            os.environ.pop("COLUMNS", None)
            os.environ.pop("LINES", None)
            for inst in list(getattr(_tqdm_std.tqdm, "_instances", [])):
                # One bar that cannot be redrawn must not cost the others their
                # repaint, so each is isolated rather than the loop as a whole.
                try:
                    if inst.disable:
                        continue
                    pos = abs(inst.pos)
                    inst.moveto(pos)
                    # tqdm's own clear() blanks the line by writing as many
                    # spaces as the *old* width; once the terminal has shrunk
                    # that padding wraps and walks the bar down a row per
                    # resize. Erase to end of line, then drop the status
                    # printer so it stops padding to the pre-resize length.
                    inst.fp.write("\r\x1b[K")
                    inst.moveto(-pos)
                    inst.sp = inst.status_printer(inst.fp)
                    inst.refresh(nolock=True)
                except Exception:
                    continue
            if callable(_prev_handler):
                _prev_handler(signum, frame)

        signal.signal(signal.SIGWINCH, _on_winch)
        _tqdm_std.tqdm._tqdm_winch_installed = True
    except (ValueError, OSError):
        # signal.signal raises ValueError off the main thread; ignore.
        pass


_harden_tqdm_against_resize()

STAGES = (
    "annotate", "segment", "transfer", "tile",
    "split", "train", "validate", "export", "report",
)
