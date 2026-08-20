import sys
from pathlib import Path

R = Path("/workspace/wsinsight/wsinsight-model-development")
sys.path.insert(0, str(R / "wsinsight-train"))
from wsitrain.tuning import _per_class_f1  # noqa: E402
import yaml  # noqa: E402

run = Path(sys.argv[1])
lm = yaml.safe_load((R / "models/trainingset/pantissue/label_map.yaml").read_text())
res = _per_class_f1(run)
if res is None:
    print(f"no val_results under {run}")
    raise SystemExit(1)
classes, f1 = res
print(f"{'class':30s} {'F1':>6s}")
for c, v in sorted(zip(classes, f1), key=lambda t: t[1]):
    print(f"{lm.get(int(c), c):30s} {v:6.3f}")
print(f"\nmacro-F1 over {len(classes)} evaluated classes: {f1.mean():.4f}")
