"""kurtorank — pan-tissue ensemble subtype annotation for gene-limited spatial transcriptomics."""
__version__ = "3.0.0"


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
