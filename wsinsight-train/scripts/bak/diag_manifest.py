import json
from pathlib import Path

R = Path("/workspace/wsinsight/wsinsight-model-development")
m = json.loads((R / "models/manifest.json").read_text())
for st in ("transfer", "tile", "split", "train"):
    s = m["stages"].get(st, {})
    info = s.get("info", s)
    if st == "transfer":
        info = {k: v for k, v in info.items()
                if k not in ("match_rate", "cells_per_sample")}
    print(f"--- {st}: {s.get('status')}")
    print({k: v for k, v in info.items() if k != "weights"})
w = m["stages"]["split"].get("info", {}).get("weights")
if w:
    print("\nweights:", [round(float(x), 2) for x in w])
print("\nconfig:", {k: v for k, v in m.get("config", {}).items()
                    if k in ("by_slide", "val_frac", "task", "transform",
                             "match_radius_px", "min_match_rate", "tune")})
