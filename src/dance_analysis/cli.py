"""Command-line entrypoint: ingest | pose | beats | compare."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import config, store
from .align import mirror_track, window_track
from .audio import BeatGrid, extract_beats
from .ingest import ingest
from .metrics import (groove_timing, moments, picture_catching, profile_track,
                      sync_between)
from .pose import PoseSequence, extract_pose
from .report import build_report
from .visualize import plot_comparison, save_track_preview


def cmd_ingest(a):
    name = config.make_clip_id(a.date, a.title, a.instructor)
    out = ingest(name, drive=a.drive, youtube=a.youtube)
    config.save_meta(name, link=a.drive or a.youtube, title=a.title,
                     instructor=a.instructor, video_date=a.date)
    print(f"id: {name}")
    print(f"saved -> {out}")
    print(f"next: python -m dance_analysis prep {name}")


def cmd_pose(a):
    video = config.raw_path(a.name)
    if not video.exists():
        sys.exit(f"no clip at {video}. Run `ingest` first or drop the mp4 there.")
    seq = extract_pose(video, name=a.name, conf=a.conf)
    print(f"pose saved -> {config.pose_path(a.name)}")
    print("\ntracks found (id, frames, seconds visible):")
    for tid, n, secs in seq.summary():
        print(f"  track {tid:>3}  {n:>5} frames  {secs:5.1f}s")
    preview = config.report_dir(a.name) / "tracks_preview.jpg"
    save_track_preview(video, seq, preview)
    print(f"\nlabeled preview -> {preview}")
    print("Open it to see which track number is you and which is the reference.")


def cmd_beats(a):
    video = config.raw_path(a.name)
    grid = extract_beats(video, name=a.name)
    print(f"tempo ~{grid.tempo:.1f} BPM, {len(grid.beats)} beats over {grid.duration:.1f}s")
    print(f"beats saved -> {config.beats_path(a.name)}")


def cmd_prep(a):
    """One shot: pose + beats + labeled preview. Then look at the preview and run compare."""
    cmd_pose(a)
    print()
    cmd_beats(a)
    print("\nNext: open the preview, note your track id and the reference's, then run:")
    print(f"  python -m dance_analysis compare {a.name} --me <id> --ref <id> --team <team>")


def _collect_feedback(a) -> list[str]:
    """General feedback (all clips) + an optional team/clip feedback file + inline."""
    feedback = []
    gf = config.general_feedback_path()
    if gf.exists():
        feedback += [ln.strip() for ln in gf.read_text().splitlines() if ln.strip()]
    if a.feedback_file:
        fp = (config.PROCESSED / a.feedback_file
              if not a.feedback_file.startswith("/") else Path(a.feedback_file))
        if fp.exists():
            feedback += [ln.strip() for ln in fp.read_text().splitlines() if ln.strip()]
    feedback += list(a.feedback or [])
    return feedback


MIN_TRACK_FRAMES = 15   # ~0.5s; below this a track is too fragmented to trust


def cmd_compare(a):
    seq = PoseSequence.load(config.pose_path(a.name))
    grid = BeatGrid.load(config.beats_path(a.name))
    if len(seq.tracks) < 2:
        sys.exit(f"need at least 2 tracked dancers; found {len(seq.tracks)}. "
                 "The footage may be too low-quality or have too few people.")
    if a.me == a.ref:
        sys.exit("--me and --ref must be different dancers.")
    if a.me not in seq.tracks or a.ref not in seq.tracks:
        sys.exit(f"track ids {a.me}/{a.ref} not found. Available: {list(seq.tracks)}")
    for who, tid in (("--me", a.me), ("--ref", a.ref)):
        if len(seq.tracks[tid].frames) < MIN_TRACK_FRAMES:
            sys.exit(f"{who} track {tid} is only {len(seq.tracks[tid].frames)} frames "
                     "(too fragmented to compare). Pick a track with more time visible.")

    me_t, ref_t = seq.tracks[a.me], seq.tracks[a.ref]
    if a.start is not None or a.end is not None:
        me_t = window_track(me_t, seq.fps, a.start, a.end)
        ref_t = window_track(ref_t, seq.fps, a.start, a.end)
    if a.mirror:
        me_t = mirror_track(me_t, seq.size[0])

    me = profile_track(me_t, seq.fps, grid)
    ref = profile_track(ref_t, seq.fps, grid)
    sync = sync_between(me_t, ref_t, seq.fps)

    feedback = _collect_feedback(a)
    meta = config.save_meta(a.name, me_desc=a.me_desc, ref_desc=a.ref_desc, team=a.team)

    out_dir = config.report_dir(a.name)
    plot = plot_comparison(me_t, ref_t, seq.fps, grid, out_dir / "motion_energy.png")
    mom = moments(me_t, ref_t, seq.fps)
    pair = {"picture_catching": picture_catching(me_t, ref_t, seq.fps),
            "groove_timing": groove_timing(me_t, ref_t, seq.fps)}
    report = build_report(a.name, me, ref, sync, [plot], team=a.team,
                          feedback=feedback, meta=meta, moments=mom, pair=pair)
    print(f"report  -> {report}")
    print(f"csv     -> {out_dir / 'metrics.csv'}")
    print(f"journal -> {out_dir / 'journal_entry.md'}  (paste into your Notion log)")


def cmd_export(a):
    out, n = store.export_all()
    print(f"exported {n} clips -> {out}")
    print("one row per clip; you_/ref_/diff_ per metric; sorted by film date.")


def main(argv=None):
    p = argparse.ArgumentParser(prog="dance_analysis",
                                description="Hip-hop self-vs-team movement analysis.")
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("ingest", help="download a clip; id = date-title-instructor")
    pi.add_argument("--date", required=True, help="film date, e.g. 2026-06-15")
    pi.add_argument("--title", required=True, help="song title")
    pi.add_argument("--instructor", required=True, help="instructor name")
    pi.add_argument("--drive", help="Google Drive share url or file id")
    pi.add_argument("--youtube", help="YouTube url")
    pi.set_defaults(func=cmd_ingest)

    pp = sub.add_parser("pose", help="extract tracked pose from data/raw/<name>.mp4")
    pp.add_argument("name")
    pp.add_argument("--conf", type=float, default=0.5)
    pp.set_defaults(func=cmd_pose)

    pb = sub.add_parser("beats", help="extract beat grid")
    pb.add_argument("name")
    pb.set_defaults(func=cmd_beats)

    pr = sub.add_parser("prep", help="pose + beats + preview in one step")
    pr.add_argument("name")
    pr.add_argument("--conf", type=float, default=0.5)
    pr.set_defaults(func=cmd_prep)

    pc = sub.add_parser("compare", help="compare against a dancer in the clip or a team profile")
    pc.add_argument("name")
    pc.add_argument("--me", type=int, required=True, help="your track id")
    pc.add_argument("--ref", type=int, required=True, help="reference dancer track id")
    pc.add_argument("--team", help="target team tag (selects style weights, labels report)")
    pc.add_argument("--me-desc", dest="me_desc", help="describe yourself in the video")
    pc.add_argument("--ref-desc", dest="ref_desc", help="describe the reference dancer")
    pc.add_argument("--start", type=float, help="window start in seconds (e.g. a unison part)")
    pc.add_argument("--end", type=float, help="window end in seconds")
    pc.add_argument("--mirror", action="store_true",
                    help="mirror your track (for mirrored formations)")
    pc.add_argument("--feedback", action="append",
                    help="a subjective note to include (repeatable)")
    pc.add_argument("--feedback-file", dest="feedback_file",
                    help="text file of feedback notes, one per line "
                         "(name only = looked up in data/processed/)")
    pc.set_defaults(func=cmd_compare)

    pe = sub.add_parser("export", help="export all clips' metrics to one CSV (progress)")
    pe.set_defaults(func=cmd_export)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
