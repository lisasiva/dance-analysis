"""Per-clip CSV export of the metrics (you vs. reference)."""

from __future__ import annotations

import csv
from pathlib import Path

from . import config

# Flat numeric metrics logged + compared (top-level + rom_region.*).
SCALARS = [
    "sharpness", "fluidity", "dynamic_range", "groove_strength", "groove_beat_lock",
    "articulation", "engagement", "pocket_ms", "pocket_std_ms",
    "timing_bias_ms", "timing_std_ms",
]
ROM_REGIONS = ["head", "shoulders", "chest", "hips", "arms", "legs"]


def flatten(profile: dict) -> dict:
    """Profile dict -> flat {metric: value} (rom_region.X expanded)."""
    out = {k: float(profile.get(k, 0) or 0) for k in SCALARS}
    reg = profile.get("rom_region", {})
    for r in ROM_REGIONS:
        out[f"rom_{r}"] = float(reg.get(r, 0) or 0)
    return out


# ---- CSV outputs ------------------------------------------------------------
def write_clip_csv(name: str, me: dict, ref: dict) -> Path:
    """Per-clip you-vs-reference table, easy to paste into another Claude session."""
    out = config.report_dir(name) / "metrics.csv"
    mf, rf = flatten(me), flatten(ref)
    with out.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["metric", "you", "reference"])
        for k in mf:
            w.writerow([k, round(mf[k], 3), round(rf[k], 3)])
    return out
