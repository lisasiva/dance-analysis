"""Local web UI for dance-analysis.

Run with:  ./run.sh   (or:  streamlit run app.py)

Non-technical flow: paste a Drive/YouTube link, enter film date / song / instructor,
describe yourself + the reference, pick the team, click through two buttons.
"""

from pathlib import Path

import streamlit as st

from dance_analysis import config, store
from dance_analysis.align import window_track
from dance_analysis.audio import BeatGrid, extract_beats
from dance_analysis.ingest import ingest
from dance_analysis.metrics import (groove_timing, moments, picture_catching,
                                    profile_track, sync_between)
from dance_analysis.pose import PoseSequence, extract_pose
from dance_analysis.report import build_report
from dance_analysis.visualize import plot_comparison, save_track_preview

st.set_page_config(page_title="Dance Analysis", page_icon="💃", layout="wide")
st.title("💃 Dance Analysis")
st.caption("Compare your performance to another dancer who's killing it, and spot where "
           "to focus next.")

DEFINITIONS = {
    "sharpness (tick)": "How crisply you hit and freeze — explosive attack, dead stops. "
                        "Higher = sharper.",
    "fluidity (gooey)": "Sustained, continuous movement vs. hit-and-freeze. Typical speed ÷ "
                        "peak speed; higher = smoother flow.",
    "pocket": "When you INITIATE a move relative to the beat. After the beat = patient / in "
              "the pocket; before = anticipating / rushing. Rough (±~33 ms).",
    "timing": "Where your accents land vs. the reference, and how consistent (lower spread = "
              "tighter).",
    "groove (beat-lock)": "Your bounce / weight-shift, and whether it rides the beat. "
                          "Strength = how much you bounce; beat-lock = how on-tempo it is.",
    "dynamic range": "Contrast between your biggest moves and your stillest moments. Higher = "
                     "more dynamic; flatness reads as 'not ready'.",
    "articulation": "Share of your motion carried by head + chest (vs. only limbs).",
    "body engagement": "How many body segments (head/chest/hips/arms/legs) move at the SAME "
                       "time in a move, 0–5. Measures 'filling up' a move.",
    "range of motion (ROM)": "How far each body region travels, in degrees — your line & "
                             "extension, per part. Turn/spin frames are excluded.",
    "sync": "How closely you match the reference's exact shape. Low priority — your own style "
            "is fine/preferred.",
}


def _report_dirs():
    dirs = [d for d in config.REPORTS.iterdir()
            if d.is_dir() and (d / "report.md").exists()]
    # newest film date first; fall back to file mtime when a date is missing
    def _key(d):
        m = config.load_meta(d.name)
        return (m.get("video_date") or "", (d / "report.md").stat().st_mtime)
    return sorted(dirs, key=_key, reverse=True)


def _report_label(d) -> str:
    m = config.load_meta(d.name)
    parts = [m.get("title") or d.name]
    if m.get("instructor"):
        parts.append(m["instructor"])
    if m.get("video_date"):
        parts.append(m["video_date"])
    return "  ·  ".join(parts)


# ---- Sidebar: general feedback (applies across all clips) --------------------
with st.sidebar:
    st.header("Your general feedback")
    st.caption("Notes that apply to you across ALL clips (not video-specific).")
    gf_path = config.general_feedback_path()
    gf_current = gf_path.read_text() if gf_path.exists() else ""
    gf_new = st.text_area("One note per line", value=gf_current, key="general_fb")
    if st.button("Save general feedback"):
        gf_path.write_text(gf_new)
        st.success("Saved — included in every future report.")

    st.divider()
    with st.expander("📖 What we measure (and how)"):
        for term, desc in DEFINITIONS.items():
            st.markdown(f"**{term}** — {desc}")

tab_analyze, tab_past = st.tabs(["Analyze", "Past reports"])

# ============================================================ ANALYZE
with tab_analyze:
    with st.form("inputs"):
        st.subheader("1 · Your clip")
        link = st.text_input("Google Drive or YouTube link",
                             help="Drive: set sharing to 'Anyone with the link'. "
                                  "YouTube: works best with a single continuous shot, "
                                  "stable camera, dancers clearly visible.")
        c1, c2, c3 = st.columns(3)
        film_date = c1.text_input("Film date", placeholder="2026-06-15")
        title = c2.text_input("Song title", placeholder="e.g. Pony")
        instructor = c3.text_input("Instructor", placeholder="e.g. Dalia")
        team = st.selectbox("Reference's team (optional, tunes emphasis)",
                            ["project-a", "fine-lines", "(generic)"])
        d1, d2 = st.columns(2)
        me_desc = d1.text_input("Describe YOURSELF in the video",
                                placeholder="e.g. gray shirt, black pants")
        ref_desc = d2.text_input("Describe the REFERENCE dancer",
                                 placeholder="e.g. all black, blonde hair")
        feedback_text = st.text_area(
            "Feedback you've received (optional, one note per line)",
            placeholder="e.g. director said be more patient\nwork on fuller arms")
        go = st.form_submit_button("Download & detect dancers")

    if go:
        if not (link and film_date and title and instructor):
            st.error("Link, film date, song title, and instructor are all required.")
        else:
            name = config.make_clip_id(film_date, title, instructor)
            is_yt = "youtube.com" in link or "youtu.be" in link
            try:
                with st.spinner("Downloading and detecting poses — a few minutes…"):
                    video = ingest(name, youtube=link) if is_yt else ingest(name, drive=link)
                    seq = extract_pose(video, name=name)
                    extract_beats(video, name=name)
                    preview = config.report_dir(name) / "tracks_preview.jpg"
                    save_track_preview(video, seq, preview)
                    config.save_meta(name, link=link, title=title, instructor=instructor,
                                     video_date=film_date, me_desc=me_desc,
                                     ref_desc=ref_desc, team=team)
            except Exception as e:  # noqa: BLE001 — surface any pipeline failure cleanly
                st.error(f"Couldn't process this clip: {e}\n\nCommon causes: the link "
                         "isn't public, the video has no audio, or ffmpeg/yt-dlp is missing.")
                st.stop()
            if len(seq.tracks) < 2:
                st.error(f"Only {len(seq.tracks)} dancer(s) detected — need at least 2 "
                         "(you + a reference). Try clearer footage with both dancers in frame.")
                st.stop()
            st.session_state["ready"] = dict(
                name=name, team=team, me_desc=me_desc, ref_desc=ref_desc,
                feedback=feedback_text, preview=str(preview), tracks=seq.summary())
            st.caption(f"Report id: `{name}`")

    state = st.session_state.get("ready")
    if state:
        st.subheader("2 · Who is who?")
        st.image(state["preview"],
                 caption="Each detected dancer is labeled with a track number.")
        if state["me_desc"]:
            st.markdown(f"You said **you** are: _{state['me_desc']}_  ·  "
                        f"**reference**: _{state['ref_desc']}_")

        opts = [f"track {tid}  ({secs:.0f}s visible)" for tid, _, secs in state["tracks"]]
        ids = [tid for tid, _, _ in state["tracks"]]
        c1, c2 = st.columns(2)
        me_sel = c1.selectbox("Which track is YOU?", opts, index=0)
        ref_sel = c2.selectbox("Which is the REFERENCE?", opts, index=min(1, len(opts) - 1))
        t1, t2 = st.columns(2)
        start_s = t1.text_input("Dance starts at (seconds, optional)",
                                help="Skip the intro freestyle/shuffle so it doesn't "
                                     "count. Leave blank to use the whole clip.")
        end_s = t2.text_input("Dance ends at (seconds, optional)",
                              help="Skip the outro. Leave blank for the whole clip.")

        if st.button("Compare"):
            me_id, ref_id = ids[opts.index(me_sel)], ids[opts.index(ref_sel)]
            if me_id == ref_id:
                st.error("Pick two different dancers for YOU and the REFERENCE.")
                st.stop()
            name = state["name"]
            seq = PoseSequence.load(config.pose_path(name))
            grid = BeatGrid.load(config.beats_path(name))
            team = None if state["team"] == "(generic)" else state["team"]
            start = float(start_s) if start_s.strip() else None
            end = float(end_s) if end_s.strip() else None
            me_t, ref_t = seq.tracks[me_id], seq.tracks[ref_id]
            if start is not None or end is not None:
                me_t = window_track(me_t, seq.fps, start, end)
                ref_t = window_track(ref_t, seq.fps, start, end)
            with st.spinner("Comparing…"):
                me = profile_track(me_t, seq.fps, grid)
                ref = profile_track(ref_t, seq.fps, grid)
                sync = sync_between(me_t, ref_t, seq.fps)
                out = config.report_dir(name)
                plot = plot_comparison(me_t, ref_t, seq.fps, grid,
                                       out / "motion_energy.png")
                mom = moments(me_t, ref_t, seq.fps)
                fb = [ln.strip() for ln in state.get("feedback", "").splitlines()
                      if ln.strip()]
                gfp = config.general_feedback_path()
                if gfp.exists():
                    fb += [ln.strip() for ln in gfp.read_text().splitlines() if ln.strip()]
                pair = {"picture_catching": picture_catching(me_t, ref_t, seq.fps),
                        "groove_timing": groove_timing(me_t, ref_t, seq.fps)}
                report = build_report(name, me, ref, sync, [plot], team=team, feedback=fb,
                                      meta=config.load_meta(name), moments=mom, pair=pair)
            st.success("Done! Also saved under the Past reports tab.")
            st.image(str(out / "motion_energy.png"))
            st.markdown(Path(report).read_text())
            csv_path = out / "metrics.csv"
            if csv_path.exists():
                st.download_button("⬇ Download metrics (CSV)", csv_path.read_bytes(),
                                   f"{name}_metrics.csv", "text/csv", key="fresh_csv")

# ============================================================ PAST REPORTS
with tab_past:
    dirs = _report_dirs()
    if not dirs:
        st.caption("No analyses yet — run one in the Analyze tab.")
    else:
        out, n = store.export_all()
        st.download_button(f"⬇ Export ALL {n} clips (CSV — track progress)",
                           out.read_bytes(), "all_metrics.csv", "text/csv",
                           help="One row per clip, every metric as you/ref/diff, "
                                "sorted by film date. Filter to a song to see the "
                                "diff shrink over time.")
        st.divider()
    for d in dirs:
        with st.expander(_report_label(d)):   # collapsed by default → compact list
            plot = d / "motion_energy.png"
            if plot.exists():
                st.image(str(plot))
            cols = st.columns(2)
            csv_path = d / "metrics.csv"
            if csv_path.exists():
                cols[0].download_button("⬇ Metrics (CSV)", csv_path.read_bytes(),
                                        f"{d.name}_metrics.csv", "text/csv",
                                        key=f"csv_{d.name}")
            cols[1].download_button("⬇ Report (Markdown)", (d / "report.md").read_bytes(),
                                    f"{d.name}_report.md", "text/markdown",
                                    key=f"md_{d.name}")
            st.markdown((d / "report.md").read_text())
