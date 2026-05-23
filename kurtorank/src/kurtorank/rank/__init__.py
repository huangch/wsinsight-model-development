"""Marker reranking against the CELLxGENE Census atlas."""


def rerank_markers(*args, **kwargs):
    """Lazy re-export of :func:`kurtorank.rank.main.rerank_markers`."""
    from kurtorank.rank.main import rerank_markers as _impl
    return _impl(*args, **kwargs)


__all__ = ["rerank_markers"]
