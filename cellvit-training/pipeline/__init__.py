"""``pipeline`` — unified CLI front-end for the cellvit-training chain.

End-to-end (Xenium → trained CellViT head):

    panel       kurtorank annotate per Xenium sample
    detect      QuPath: QuST tissue + StarDist + Xenium cluster transfer
    label       QuPath: cluster_id -> label-name remap
    export      QuPath: detections -> tile PNG + per-tile CSV
    prepare     splits + class weights + fold YAML  (native Python)
    train       train classifier head + TorchScript export
    validate    confusion + classification report
    audit       split-reuse / leakage check
    aggregate   build trainingset/pantissue/ from per-tissue splits
    orchestrate outer-loop agent that retunes weights / lr / drop_rate
    run         end-to-end driver for one tissue (--from / --to)

Design rules:

1.  One ``paths.py`` is the only source of truth for directory layout, the
    QuPath project location, the Xenium data root, the CellViT runtime,
    and the resolution of ``QuPath`` / ``kurtorank`` / ``python``.
2.  v2 wraps battle-tested v1 steps via ``subprocess`` rather than rewriting
    them; Groovy stays Groovy, the bash launchers stay bash, kurtorank stays
    the canonical kurtorank CLI. Only the ``prepare`` step is implemented
    natively in Python because it already was three Python scripts.
3.  Subcommands are uniformly ``add_arguments(parser)`` + ``run(args)``.
4.  Old ``pipeline/`` is left in place and remains authoritative until each
    v2 subcommand has been verified side-by-side.
"""
from __future__ import annotations

__all__ = ["cli"]
