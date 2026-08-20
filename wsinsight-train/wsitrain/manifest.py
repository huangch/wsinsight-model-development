"""Per-run manifest: provenance + per-stage status, enabling visible resume."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import STAGES

# Config key -> earliest stage whose output stops being valid when the key changes.
# Everything from that stage onward is re-run. Perf-only knobs (gpus, batch sizes)
# are deliberately absent.
_INVALIDATES: dict[str, str] = {
    "input": "annotate",
    "task": "annotate",
    "markers_csv": "annotate",
    "top_k_markers": "annotate",
    "segmenter": "segment",
    "cellpose_model": "segment",
    "diameter": "segment",
    "cellpose_flow_threshold": "segment",
    # mpp sets the segmentation rescale factor, not just the tile geometry.
    "mpp": "segment",
    "transform": "transfer",
    "match_radius_px": "transfer",
    "min_match_rate": "transfer",
    "drop_labels": "transfer",
    "tile_px": "tile",
    "min_cells": "tile",
    "bg_thresh": "tile",
    "overlap": "tile",
    "val_frac": "split",
    "by_slide": "split",
    "seed": "split",
    "weight_cap": "split",
    "backbone": "train",
    "fold": "train",
    "tune": "train",
}


@dataclass
class Manifest:
    path: Path
    config: dict[str, Any] = field(default_factory=dict)
    stages: dict[str, dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def load_or_new(cls, path: Path, config: dict[str, Any]) -> "Manifest":
        path = Path(path)
        if not path.exists():
            return cls(path=path, config=config)
        d = json.loads(path.read_text())
        mf = cls(path=path, config=d.get("config", {}), stages=d.get("stages", {}))
        stale = mf._stale_stages(config)
        mf.config = config
        if stale:
            for stage in stale:
                mf.stages.pop(stage, None)
            mf.save()
        return mf

    def _stale_stages(self, new_config: dict[str, Any]) -> list[str]:
        earliest: str | None = None
        changed: list[str] = []
        for key, stage in _INVALIDATES.items():
            if key not in self.config or self.config[key] == new_config.get(key):
                continue
            changed.append(f"{key}: {self.config[key]!r} -> {new_config.get(key)!r}")
            if earliest is None or STAGES.index(stage) < STAGES.index(earliest):
                earliest = stage
        if earliest is None:
            return []
        stale = [s for s in STAGES[STAGES.index(earliest):] if self.is_done(s)]
        if stale:
            print(f"[manifest] config changed ({'; '.join(changed)}) — "
                  f"invalidating: {stale}")
        return stale

    def is_done(self, stage: str) -> bool:
        return self.stages.get(stage, {}).get("status") == "done"

    def mark(self, stage: str, status: str, **info: Any) -> None:
        self.stages[stage] = {"status": status, "ts": time.time(), **info}
        self.save()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(
            {"config": self.config, "stages": self.stages}, indent=2))
