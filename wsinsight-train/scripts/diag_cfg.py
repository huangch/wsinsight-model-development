from pathlib import Path

from wsitrain.config import build_config

c = build_config(Path("data/xenium"), "pantissue", Path("models/pannuke"),
                 overrides={"task": "pannuke"})
print("task", c.task, "| drop_labels", c.drop_labels,
      "| transform", c.transform, "| radius", c.match_radius_px,
      "| by_slide", c.by_slide)
