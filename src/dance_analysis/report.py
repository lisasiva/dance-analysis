"""Turn metric profiles into a ranked gap report."""

from __future__ import annotations

import datetime
import json
from pathlib import Path

from . import config, store


def _gap_scores(me: dict, ref: dict, sync: dict, team: str | None) -> dict:
    """Normalized 0..1 gap per dimension (1 = biggest gap to close)."""

    def rel(a, b):  # how far `me` (a) is below target `b`, normalized
        if not b:
            return 0.0
        return max(0.0, (b - a) / abs(b))

    # Timing is scored RELATIVE to the reference dancer in the SAME clip. This
    # cancels the beat-detector's own offset (absolute 'pocket' isn't reliably
    # measurable), and is the only statistically valid timing comparison.
    # NOTE: still confounded by sharpness — softer (broader) accents read later.
    timing_gap = min(1.0, abs(me["timing_bias_ms"] - ref["timing_bias_ms"]) / 100.0)

    sharp_gap = min(1.0, rel(me["sharpness"], ref["sharpness"]))
    fluid_gap = min(1.0, rel(me["fluidity"], ref["fluidity"]))
    dyn_gap = min(1.0, rel(me["dynamic_range"], ref["dynamic_range"]))

    # groove = oscillation strength weighted by how well it locks to the beat
    me_groove = me["groove_strength"] * me["groove_beat_lock"]
    ref_groove = ref["groove_strength"] * ref["groove_beat_lock"]
    groove_gap = min(1.0, rel(me_groove, ref_groove))

    # engagement = body parts moving per move ("filling up" movement)
    eng_gap = min(1.0, rel(me["engagement"], ref["engagement"]))

    angle_diff = sync.get("angle_diff_deg") or 0.0
    sync_gap = min(1.0, angle_diff / 30.0)

    imb = me.get("symmetry_imbalance", {})
    sym_gap = max(imb.values()) if imb else 0.0
    # region ROM shortfall vs the reference, per body part
    me_reg, ref_reg = me.get("rom_region", {}), ref.get("rom_region", {})
    reg_gap = max((rel(me_reg[k], ref_reg.get(k, 0)) for k in me_reg), default=0.0)
    rom_gap = min(1.0, max(sym_gap, reg_gap))

    return {
        "timing": timing_gap, "sharpness": sharp_gap, "fluidity": fluid_gap,
        "dynamics": dyn_gap, "groove": groove_gap, "engagement": eng_gap,
        "sync": sync_gap, "rom": rom_gap,
    }


def _verdict(me_v: float, ref_v: float, deadband: float = 0.12) -> str:
    """Compare a value to the reference. The reference IS the target, so within
    the deadband = matched (success); outside = a real difference to look at."""
    if not ref_v:
        return "—"
    rel = (me_v - ref_v) / abs(ref_v)
    pct, dirn = abs(rel) * 100, "below" if rel < 0 else "above"
    if abs(rel) <= deadband:
        return f"✓ matched ({pct:.0f}% {dirn})" if pct >= 1 else "✓ matched"
    return f"{'▼' if rel < 0 else '▲'} {pct:.0f}% {dirn}"


def _compare_table(me: dict, ref: dict) -> list[str]:
    """A flat 'you vs reference' table with a matched/off verdict per metric, so
    it's obvious what's at-reference vs. what's a genuine gap."""
    g = me["groove_strength"] * me["groove_beat_lock"]
    gr = ref["groove_strength"] * ref["groove_beat_lock"]
    rows = [
        ("sharpness (tick)", f"{me['sharpness']:.1f}", f"{ref['sharpness']:.1f}",
         _verdict(me["sharpness"], ref["sharpness"])),
        ("fluidity (gooey)", f"{me['fluidity']:.2f}", f"{ref['fluidity']:.2f}",
         _verdict(me["fluidity"], ref["fluidity"])),
        ("dynamic range", f"{me['dynamic_range']:.2f}", f"{ref['dynamic_range']:.2f}",
         _verdict(me["dynamic_range"], ref["dynamic_range"])),
        ("groove (locked bounce)", f"{g:.3f}", f"{gr:.3f}", _verdict(g, gr, 0.25)),
        ("articulation (head+chest)", f"{me['articulation']:.2f}",
         f"{ref['articulation']:.2f}", _verdict(me["articulation"], ref["articulation"])),
        ("body engagement (parts/move, of 5)", f"{me['engagement']:.1f}",
         f"{ref['engagement']:.1f}", _verdict(me["engagement"], ref["engagement"])),
    ]
    for k in ("head", "shoulders", "chest", "hips", "arms", "legs"):
        mv, rv = me.get("rom_region", {}).get(k, 0), ref.get("rom_region", {}).get(k, 0)
        rows.append((f"ROM · {k}", f"{mv:.0f}°", f"{rv:.0f}°", _verdict(mv, rv)))

    out = ["\n## You vs. reference (at a glance)\n",
           "Goal is to **match** the reference. `✓ matched` = you're already there.\n",
           "| Metric | You | Ref | Verdict |", "|---|---|---|---|"]
    for name_, mv, rv, v in rows:
        out.append(f"| {name_} | {mv} | {rv} | {v} |")
    out.append("\n*Differences inside ~12% read as matched (within measurement noise). "
               "A difference is only trustworthy if it repeats across several reference clips.*")
    return out


# key metrics for strengths/weaknesses + the side-by-side (label, accessor, higher=better)
_KEY_METRICS = [
    ("sharpness (tick)", lambda m: m["sharpness"]),
    ("fluidity (gooey)", lambda m: m["fluidity"]),
    ("dynamic range", lambda m: m["dynamic_range"]),
    ("groove (locked bounce)", lambda m: m["groove_strength"] * m["groove_beat_lock"]),
    ("articulation (head+chest)", lambda m: m["articulation"]),
    ("body engagement (parts/move)", lambda m: m["engagement"]),
    ("ROM head", lambda m: m.get("rom_region", {}).get("head", 0)),
    ("ROM shoulders", lambda m: m.get("rom_region", {}).get("shoulders", 0)),
    ("ROM chest", lambda m: m.get("rom_region", {}).get("chest", 0)),
    ("ROM hips", lambda m: m.get("rom_region", {}).get("hips", 0)),
    ("ROM arms", lambda m: m.get("rom_region", {}).get("arms", 0)),
    ("ROM legs", lambda m: m.get("rom_region", {}).get("legs", 0)),
]


def _classify(me: dict, ref: dict, band: float = 0.12):
    """Split key metrics into strengths (>band above ref) and weaknesses (>band
    below ref), each sorted by how far from the reference."""
    strengths, weaknesses = [], []
    for label, fn in _KEY_METRICS:
        mv, rv = fn(me), fn(ref)
        if not rv:
            continue
        rel = (mv - rv) / abs(rv)
        if rel > band:
            strengths.append((rel, f"{label}: {rel * 100:.0f}% above reference"))
        elif rel < -band:
            weaknesses.append((-rel, f"{label}: {abs(rel) * 100:.0f}% below reference"))
    strengths.sort(reverse=True)
    weaknesses.sort(reverse=True)
    return [s for _, s in strengths], [w for _, w in weaknesses]


def _journal_entry(name: str, meta: dict, team: str | None, me: dict, ref: dict,
                   strengths: list[str], weaknesses: list[str]) -> str:
    """Paste-into-Notion summary, following the agreed spec."""
    song = meta.get("title") or meta.get("nickname") or name
    link = meta.get("link")
    g = me["groove_strength"] * me["groove_beat_lock"]
    gr = ref["groove_strength"] * ref["groove_beat_lock"]
    lines = [
        f"### {song}" + (f" — {meta['video_date']}" if meta.get("video_date") else ""),
        f"- **Song:** {song}" + (f" — [video]({link})" if link else ""),
        f"- **Instructor:** {meta.get('instructor') or '—'}  ·  "
        f"**Film date:** {meta.get('video_date') or '—'}",
        f"- **You:** {meta.get('me_desc') or '—'}",
        f"- **Reference:** {meta.get('ref_desc') or '—'}"
        + (f"  ·  team: {team}" if team else ""),
        "",
        "**Key metrics (you vs reference):**",
        "| Metric | You | Ref |",
        "|---|---|---|",
        f"| sharpness | {me['sharpness']:.1f} | {ref['sharpness']:.1f} |",
        f"| fluidity | {me['fluidity']:.2f} | {ref['fluidity']:.2f} |",
        f"| dynamic range | {me['dynamic_range']:.2f} | {ref['dynamic_range']:.2f} |",
        f"| groove (locked) | {g:.3f} | {gr:.3f} |",
        f"| articulation | {me['articulation']:.2f} | {ref['articulation']:.2f} |",
        f"| body engagement (/5) | {me['engagement']:.1f} | {ref['engagement']:.1f} |",
        f"| ROM chest | {me['rom_region']['chest']:.0f}° | {ref['rom_region']['chest']:.0f}° |",
        f"| ROM hips | {me['rom_region']['hips']:.0f}° | {ref['rom_region']['hips']:.0f}° |",
        f"| ROM arms | {me['rom_region']['arms']:.0f}° | {ref['rom_region']['arms']:.0f}° |",
        f"| ROM legs | {me['rom_region']['legs']:.0f}° | {ref['rom_region']['legs']:.0f}° |",
        "",
        "**Strengths:** " + ("; ".join(strengths) if strengths else "—"),
        "**Weaknesses:** " + ("; ".join(weaknesses) if weaknesses else "—"),
        "",
    ]
    return "\n".join(lines)


def build_report(name: str, me: dict, ref: dict, sync: dict,
                 plots: list[Path], team: str | None = None,
                 feedback: list[str] | None = None,
                 meta: dict | None = None, moments: dict | None = None) -> Path:
    meta = meta or {}
    moments = moments or {}
    strengths, weaknesses = _classify(me, ref)
    gaps = _gap_scores(me, ref, sync, team)
    weights = config.weights_for(team)
    weighted = {k: gaps[k] * weights[k] for k in gaps}
    # Sync is deprioritized: matching a reference's exact shape rewards conformity
    # over your own style, so it's shown as a number but never drives the ranking.
    weighted["sync"] = 0.0
    ranked = sorted(weighted, key=weighted.get, reverse=True)

    out_dir = config.report_dir(name)
    # raw metrics
    (out_dir / "metrics.json").write_text(json.dumps(
        {"name": name, "team": team, "me": me, "reference": ref, "sync": sync,
         "gaps": gaps, "weighted": weighted, "ranked": ranked},
        indent=2, default=float,
    ))
    # Notion-paste-friendly one-block summary
    (out_dir / "journal_entry.md").write_text(
        _journal_entry(name, meta, team, me, ref, strengths, weaknesses))

    song = meta.get("title") or meta.get("nickname") or name
    analyzed_date = datetime.date.today().isoformat()
    title = f"# Dance gap report — {song}"
    if team:
        title += f"  (target team: {team})"
    lines = [title + "\n"]
    lines.append(f"- **Song:** {song}"
                 + (f" — [video]({meta['link']})" if meta.get("link") else ""))
    lines.append(f"- **Instructor:** {meta.get('instructor') or '—'}")
    lines.append(f"- **Film date:** {meta.get('video_date') or '—'}  ·  "
                 f"**Analyzed:** {analyzed_date}")
    lines.append(f"- **You:** {meta.get('me_desc') or '—'}")
    lines.append(f"- **Reference:** {meta.get('ref_desc') or '—'}"
                 + (f"  ·  team: {team}" if team else ""))
    lines.append(f"\n- **Strengths:** {'; '.join(strengths) if strengths else '—'}")
    lines.append(f"- **Weaknesses:** {'; '.join(weaknesses) if weaknesses else '—'}")
    lines.append("## Where to focus (ranked)\n")
    lines.append("| Rank | Dimension | Gap (0–1) | What it means |")
    lines.append("|---|---|---|---|")
    meaning = {
        "timing": "accent placement vs. the reference (relative)",
        "sharpness": "attack/freeze crispness (tick)",
        "fluidity": "sustained, gooey movement",
        "dynamics": "contrast between big moves and stillness",
        "groove": "bounce / weight-shift locked to the beat",
        "engagement": "engaging multiple body parts per move",
        "sync": "matching the reference's shape (low priority — style is yours)",
        "rom": "range of motion / line & extension",
    }
    for i, k in enumerate(ranked, 1):
        note = meaning[k]
        if k == "sync":
            note += " — not ranked (low priority; your style is yours)"
        lines.append(f"| {i} | {k} | {gaps[k]:.2f} | {note} |")

    lines.extend(_compare_table(me, ref))

    if moments:
        lines.append("\n## Moments to study (timestamps in this clip)\n")
        if moments.get("leg_extension"):
            lines.append(f"- **Leg extension** @ {moments['leg_extension']} — the reference "
                         "hits a much bigger stance than you here; compare your version.")
        if moments.get("fluidity_gaps"):
            ts = ", ".join(moments["fluidity_gaps"])
            lines.append(f"- **Fluidity gaps** @ {ts} — the reference keeps flowing while "
                         "you've frozen; add sustained motion through these.")
        if moments.get("groove_section"):
            lines.append(f"- **Groove section** @ {moments['groove_section']} — steady-motion "
                         "stretch; drill the weight shifts to a count here.")
        if moments.get("your_sharp_hit"):
            lines.append(f"- **Your sharpest hit** @ {moments['your_sharp_hit']} — a strength; "
                         "this is your crispness at its best.")

    # pocket: when movement initiates relative to the beat (rough, absolute-ish)
    def _pk(v):
        return f"{abs(v):.0f} ms {'after' if v >= 0 else 'before'} the beat"
    pk_rel = me["pocket_ms"] - ref["pocket_ms"]
    lines.append("\n## Pocket (when you initiate movement)\n")
    lines.append(f"- **You:** ~{_pk(me['pocket_ms'])}  ·  **Reference:** ~{_pk(ref['pocket_ms'])}")
    lines.append(f"- **Relative:** you initiate **{abs(pk_rel):.0f} ms "
                 f"{'later' if pk_rel > 0 else 'earlier'}** than the reference "
                 "(the reliable number — absolute is rough, ±~33 ms at 30fps)")
    lines.append("- *Positive = after the beat (patient / in the pocket); "
                 "negative = before the beat (anticipating / rushing).*")

    rel = me["timing_bias_ms"] - ref["timing_bias_ms"]
    lines.append("\n## Key numbers\n")
    lines.append(f"- **Timing vs. reference (same clip):** you land **{abs(rel):.0f} ms "
                 f"{'later than' if rel > 0 else 'earlier than'}** the reference "
                 f"(relative only — absolute pocket isn't measurable; also confounded "
                 f"by sharpness, so treat director feel-feedback as ground truth)")
    lines.append(f"- **Timing consistency (std):** you {me['timing_std_ms']:.0f} ms vs. "
                 f"ref {ref['timing_std_ms']:.0f} ms (lower = tighter)")
    lines.append(f"- **Sharpness (tick):** you {me['sharpness']:.2f} vs. ref {ref['sharpness']:.2f}")
    lines.append(f"- **Fluidity (gooey):** you {me['fluidity']:.2f} vs. ref {ref['fluidity']:.2f} "
                 f"(higher = more sustained flow)")
    lines.append(f"- **Groove:** you strength {me['groove_strength']:.3f} / beat-lock "
                 f"{me['groove_beat_lock']:.2f} vs. ref {ref['groove_strength']:.3f} / "
                 f"{ref['groove_beat_lock']:.2f}")
    lines.append(f"- **Articulation (head+chest share):** you {me['articulation']:.2f} vs. "
                 f"ref {ref['articulation']:.2f}")
    lines.append(f"- **Dynamic range:** you {me['dynamic_range']:.3f} vs. ref "
                 f"{ref['dynamic_range']:.3f}")
    me_reg, ref_reg = me.get("rom_region", {}), ref.get("rom_region", {})
    if me_reg:
        lines.append("\n### Range of motion by body part (degrees)\n")
        lines.append("| Region | You | Ref | You vs. ref |")
        lines.append("|---|---|---|---|")
        for k in ("head", "shoulders", "chest", "hips", "arms", "legs"):
            mv, rv = me_reg.get(k, 0), ref_reg.get(k, 0)
            flag = "▼ smaller" if rv and mv < 0.9 * rv else (
                "▲ bigger" if rv and mv > 1.1 * rv else "≈")
            lines.append(f"| {k} | {mv:.0f}° | {rv:.0f}° | {flag} |")
    if sync.get("angle_diff_deg") is not None:
        lines.append(f"- **Mean pose difference:** {sync['angle_diff_deg']:.1f}° "
                     f"(sync lag {sync.get('lag_ms', 0):+.0f} ms) — only meaningful on unison clips")

    if feedback:
        lines.append("\n## Subjective feedback (to reconcile against the data)\n")
        for f in feedback:
            lines.append(f"- {f}")
        lines.append("\n> Treat each as a hypothesis: where it agrees with a metric above, "
                     "prioritize it; where it conflicts, flag for a closer look.")

    lines.append("\n## Plots\n")
    for p in plots:
        lines.append(f"![{p.stem}]({p.name})")

    lines.append("\n---\n*Generated by dance-analysis.*")

    report = out_dir / "report.md"
    report.write_text("\n".join(lines))

    store.write_clip_csv(name, me, ref)   # per-clip metrics CSV
    return report
