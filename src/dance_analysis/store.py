"""Per-clip and cross-clip CSV export of the metrics (you vs. reference)."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from . import config

# Flat numeric metrics logged + compared (top-level + rom_region.*).
SCALARS = [
    "sharpness", "explosiveness", "fluidity", "dynamic_range",
    "groove_strength", "groove_beat_lock", "articulation", "engagement",
    "hip_articulation", "core_share",
    "pocket_ms", "pocket_std_ms", "timing_bias_ms", "timing_std_ms",
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


def export_all(out_path: Path | None = None) -> tuple[Path, int]:
    """One row per analyzed clip, every metric as you / ref / diff, sorted by film
    date — so you can filter to a song and watch the diffs shrink over time.

    `diff` = you − reference (signed). For most metrics, diff trending toward 0
    means you're closing the gap.
    """
    rows = []
    for d in sorted(config.REPORTS.iterdir()):
        mj = d / "metrics.json"
        if not d.is_dir() or not mj.exists():
            continue
        try:
            data = json.loads(mj.read_text())
        except json.JSONDecodeError:
            continue
        meta = config.load_meta(d.name)
        mf, rf = flatten(data.get("me", {})), flatten(data.get("reference", {}))
        row = {
            "clip": d.name,
            "film_date": meta.get("video_date", ""),
            "song": meta.get("title", ""),
            "instructor": meta.get("instructor", ""),
            "team": data.get("team") or "",
        }
        for k in mf:
            row[f"you_{k}"] = round(mf[k], 3)
            row[f"ref_{k}"] = round(rf[k], 3)
            row[f"diff_{k}"] = round(mf[k] - rf[k], 3)
        pair = data.get("pair") or {}
        pc, gt = pair.get("picture_catching"), pair.get("groove_timing")
        if pc:
            row["picture_corr"] = round(pc["corr"], 3)
            row["picture_ratio"] = round(pc["ratio"], 2)
        if gt:
            row["groove_lag_ms"] = round(gt["lag_ms"])
        rows.append(row)

    rows.sort(key=lambda r: (r["film_date"], r["clip"]))
    cols: list[str] = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    out = out_path or (config.DATA / "all_metrics.csv")
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    return out, len(rows)
