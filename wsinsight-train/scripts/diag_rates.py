"""Per-slide registration match rate from the last transfer stage, worst first."""
import json
import sys
from pathlib import Path

m = json.loads(Path(sys.argv[1]).read_text())
info = m["stages"]["transfer"].get("info", m["stages"]["transfer"])
rates = info["match_rate"]
dropped = set(info.get("dropped_slides", []))
print(f"{'tissue':12s} {'rate':>6s}  slide")
for sid, r in sorted(rates.items(), key=lambda kv: kv[1]):
    tis, _, name = sid.partition("__")
    flag = "DROP" if sid in dropped else ""
    print(f"{tis:12s} {r*100:5.1f}%  {flag:4s} {name}")
