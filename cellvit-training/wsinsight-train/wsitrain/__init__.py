"""wsinsight-train — headless CLI to train WSInsight CellViT heads."""

__version__ = "0.1.0"

STAGES = (
    "annotate", "segment", "transfer", "tile",
    "split", "train", "validate", "export", "report",
)
