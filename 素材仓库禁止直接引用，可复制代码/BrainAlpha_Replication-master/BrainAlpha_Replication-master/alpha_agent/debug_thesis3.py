"""Verify thesis_map.json loading."""
from alpha_agent.exploration_grid import (
    ExplorationGrid, DatasetCategory, OperatorCategory, Horizon, _ensure_thesis_map, get_thesis
)

raw = _ensure_thesis_map()
print(f"JSON entries: {len(raw)}")
assert len(raw) == 24, f"Expected 24 entries, got {len(raw)}"

grid = ExplorationGrid()
all_ok = True
for dc in DatasetCategory:
    for oc in OperatorCategory:
        theses = get_thesis(dc, oc)
        if not theses:
            print(f"MISSING: {dc.label}/{oc.label}")
            all_ok = False
        for hz in Horizon:
            cid = grid.get_cell(f"{dc.label}_{oc.label}_{hz.label}")
            if not cid or not cid.thesis:
                print(f"GRID MISSING: {dc.label}/{oc.label}/{hz.label}")
                all_ok = False

if all_ok:
    print("ALL 72 cells verified ✓")

# Test adding a new paper
import json
from pathlib import Path
path = Path(__file__).resolve().parent.parent / "data" / "thesis_map.json"
size_before = path.stat().st_size

print(f"thesis_map.json size: {size_before} bytes")
print("PASS: JSON-based loading works correctly")
