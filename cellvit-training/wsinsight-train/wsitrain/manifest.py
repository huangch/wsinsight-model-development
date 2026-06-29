"""Per-run manifest: provenance + per-stage status, enabling visible resume."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Manifest:
    path: Path
    config: dict[str, Any] = field(default_factory=dict)
    stages: dict[str, dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def load_or_new(cls, path: Path, config: dict[str, Any]) -> "Manifest":
        path = Path(path)
        if path.exists():
            d = json.loads(path.read_text())
            return cls(path=path, config=d.get("config", config),
                       stages=d.get("stages", {}))
        return cls(path=path, config=config)

    def is_done(self, stage: str) -> bool:
        return self.stages.get(stage, {}).get("status") == "done"

    def mark(self, stage: str, status: str, **info: Any) -> None:
        self.stages[stage] = {"status": status, "ts": time.time(), **info}
        self.save()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(
            {"config": self.config, "stages": self.stages}, indent=2))
