"""Seed-panel builder: fetch cell-type marker lists from external atlases.

Currently supports one source: ``disco`` (CELLxGENE-independent pan-tissue
atlas at https://immunesinglecell.com). Output is a *skeleton* CSV with
``tissue_type``, ``subtype``, ``markers``, ``source``, ``added_at``,
``n_cells``, ``logfc_min``, ``pct1_min``. The biology columns consumed by
``kurtorank annotate`` (``major_type``, ``pannuke_label``, ``hne_type``,
``hne_label``, ``common``, ``malignant``) are **not** produced by
``build-panel`` \u2014 they are hand-curated and must be filled in by the user
before feeding the panel to ``annotate``.
"""
from kurtorank.seed.main import build_panel

__all__ = ["build_panel"]
