# ============================
# PART 1/5 — ZIP Upload + Extraction + Segment Detection (PATH-ONLY)
# ============================
# app.py — UAPpress Video Creator
#
# ✅ ONE upload flow
# ✅ ZIP saved immediately to disk
# ✅ session_state stores only paths + small dicts
# ✅ NO hashing, NO manifest, NO ZIP bytes in memory
# ✅ Segments normalized for Part 3

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import List, Dict

import streamlit as st
import video_pipeline as vp

# ----------------------------
# Page setup
# ----------------------------
st.set_page_config(page_title="UAPpress — Video Creator", layout="wide")
st.title("🛸 UAPpress — Video Creator")
st.caption("Upload a TTS Studio ZIP → Generate segment MP4s (no subs, no logos, no stitching).")

# ----------------------------
# Safe session_state init
# ----------------------------
DEFAULTS = {
    "zip_path": "",        # path to uploaded zip on disk
    "workdir": "",         # temp working directory
    "extract_dir": "",     # extracted zip folder
    "segments": [],        # normalized segment list
    "last_error": "",
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ----------------------------
# Helpers
# ----------------------------
def _reset_zip_state() -> None:
    try:
        if st.session_state.workdir and os.path.isdir(st.session_state.workdir):
            shutil.rmtree(st.session_state.workdir, ignore_errors=True)
    except Exception:
        pass

    st.session_state.zip_path = ""
    st.session_state.workdir = ""
    st.session_state.extract_dir = ""
    st.session_state.segments = []
    st.session_state.last_error = ""

def _save_uploaded_zip(uploaded_file) -> str:
    """
    Save uploaded ZIP to disk immediately.
    Returns absolute file path.
    """
    root = tempfile.mkdtemp(prefix="uappress_zip_")
    zip_path = os.path.join(root, "tts_studio_upload.zip")
    with open(zip_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return zip_path

def _normalize_segments(pairs: List[dict]) -> List[dict]:
    """
    Convert vp.pair_segments output into a stable structure
    expected by Part 3.
    Enforces order: Intro → Chapters → Outro → Others
    """

    def label_of(p): 
        return vp.segment_label(p)

    intro = [p for p in pairs if label_of(p) == "INTRO"]
    outro = [p for p in pairs if label_of(p) == "OUTRO"]
    chapters = [p for p in pairs if label_of(p).startswith("CHAPTER")]
    others = [p for p in pairs if p not in intro + chapters + outro]

    def ch_key(p: dict):
        n = p.get("chapter_no")
        if n is None:
            return (999999, (p.get("title_guess") or "").lower())
        return (int(n), (p.get("title_guess") or "").lower())

    chapters = sorted(chapters, key=ch_key)
    others = sorted(others, key=lambda p: (p.get("title_guess") or "").lower())

    ordered = intro + chapters + outro + others

    segments = []
    for i, p in enumerate(ordered, start=1):
        label = label_of(p)
        title = p.get("title_guess") or ""

        if label == "INTRO":
            key = "intro"
        elif label == "OUTRO":
            key = "outro"
        elif label.startswith("CHAPTER") and p.get("chapter_no") is not None:
            key = f"chapter_{int(p['chapter_no']):02d}"
        else:
            key = f"segment_{i:02d}"

        segments.append({
            "index": i,
            "key": key,
            "label": label.title(),
            "title": title,
            "pair": p,  # original pipeline pair (audio + script paths)
        })

    return segments

# ----------------------------
# 1) Upload ZIP
# ----------------------------
st.subheader("1) Upload ZIP from TTS Studio (scripts + audio)")

uploaded = st.file_uploader(
    "TTS Studio ZIP",
    type=["zip"],
    help="ZIP must contain scripts (.txt/.md/.json) and audio (.mp3/.wav/etc).",
)

colA, colB = st.columns([1, 1])
with colA:
    if st.button("❌ Remove ZIP", disabled=(uploaded is None and not st.session_state.zip_path)):
        _reset_zip_state()
        st.rerun()

# ----------------------------
# 2) Save ZIP → Extract → Detect segments
# ----------------------------
if uploaded is not None:
    try:
        _reset_zip_state()

        zip_path = _save_uploaded_zip(uploaded)
        st.session_state.zip_path = zip_path

        # IMPORTANT:
        # extract_zip_to_temp MUST accept a ZIP FILE PATH
        workdir, extract_dir = vp.extract_zip_to_temp(zip_path)

        scripts, audios = vp.find_files(extract_dir)
        if not scripts:
            raise RuntimeError("No scripts found in ZIP.")
        if not audios:
            raise RuntimeError("No audio files found in ZIP.")

        pairs = vp.pair_segments(scripts, audios)
        segments = _normalize_segments(pairs)

        st.session_state.workdir = workdir
        st.session_state.extract_dir = extract_dir
        st.session_state.segments = segments
        st.session_state.last_error = ""

    except Exception as e:
        st.session_state.last_error = f"{type(e).__name__}: {e}"

# ----------------------------
# Status + Preview
# ----------------------------
if st.session_state.last_error:
    st.error(st.session_state.last_error)

if st.session_state.zip_path:
    st.success("ZIP uploaded and extracted successfully.")
    st.write(f"Detected **{len(st.session_state.segments)}** segment(s).")

    with st.expander("Show detected segments"):
        for s in st.session_state.segments:
            st.write(f"{s['index']}. **{s['label']}** — {s['title'] or 'Untitled'}")

st.caption("Next: Part 3 adds the ONE **Generate Videos** button (sequential, crash-safe).")
# ============================
# PART 3/5 — Generate Segment MP4s (Sequential, Crash-Safe)
# ============================

import gc
import time
from pathlib import Path
import streamlit as st
import video_pipeline as vp

# ----------------------------
# Session state init (small only)
# ----------------------------
if "is_generating" not in st.session_state:
    st.session_state["is_generating"] = False
if "stop_requested" not in st.session_state:
    st.session_state["stop_requested"] = False
if "generated" not in st.session_state:
    st.session_state["generated"] = {}  # key -> mp4_path (strings only)
if "gen_log" not in st.session_state:
    st.session_state["gen_log"] = []    # list[str]

def _log(msg: str) -> None:
    st.session_state["gen_log"].append(msg)

def _safe_mkdir(p: str) -> str:
    Path(p).mkdir(parents=True, exist_ok=True)
    return p

def _default_out_dir(extract_dir: str) -> str:
    # Save outputs inside extracted workspace
    return _safe_mkdir(str(Path(extract_dir) / "_mp4_segments"))

def _segment_out_path(out_dir: str, segment_key: str) -> str:
    return str(Path(out_dir) / f"{segment_key}.mp4")

def _request_stop() -> None:
    st.session_state["stop_requested"] = True
    _log("🛑 Stop requested — will stop after current segment finishes.")

def _reset_gen_flags() -> None:
    st.session_state["is_generating"] = False
    st.session_state["stop_requested"] = False

def generate_all_segments_sequential(
    *,
    segments: list,
    extract_dir: str,
    out_dir: str,
    overwrite: bool,
    zoom_strength: float,
    fps: int,
    width: int,
    height: int,
) -> None:
    st.session_state["is_generating"] = True
    st.session_state["stop_requested"] = False

    progress = st.progress(0.0)
    status = st.empty()
    detail = st.empty()

    n = len(segments)
    if n == 0:
        status.error("No segments detected.")
        _reset_gen_flags()
        return

    for i, seg in enumerate(segments, start=1):
        if st.session_state.get("stop_requested"):
            _log("Stopped before starting next segment.")
            break

        seg_key = seg.get("key", f"segment_{i:02d}")
        seg_label = seg.get("label", seg_key)
        out_path = _segment_out_path(out_dir, seg_key)

        # Skip if already exists (resume-safe) unless overwrite
        if (not overwrite) and Path(out_path).exists():
            st.session_state["generated"][seg_key] = out_path
            status.info(f"Skipping (already exists): {seg_label}")
            progress.progress(min(1.0, i / n))
            continue

        status.info(f"Generating {seg_label} ({i}/{n})…")
        detail.caption(f"Output: {out_path}")

        t0 = time.time()
        try:
            # ------------------------------------------------------------
            # ✅ Pipeline call — ONE SEGMENT at a time
            #
            # We pass seg["pair"] (the raw pair from vp.pair_segments),
            # because Part 1 stored it inside the normalized segment dict.
            #
            # If your function is named differently, change only this call.
            # ------------------------------------------------------------
            vp.render_segment_mp4(
                pair=seg["pair"],          # <-- raw pair from video_pipeline pairing
                extract_dir=extract_dir,
                out_path=out_path,
                ken_burns=True,
                zoom_only=True,
                zoom_strength=zoom_strength,
                fps=fps,
                width=width,
                height=height,
            )

            st.session_state["generated"][seg_key] = out_path

            dt = time.time() - t0
            _log(f"✅ Generated {seg_label} in {dt:.1f}s")

        except Exception as e:
            _log(f"❌ Failed {seg_label}: {type(e).__name__}: {e}")
            status.error(f"Failed generating {seg_label}. See log below.")
            break
        finally:
            # free memory between segments
            gc.collect()
            time.sleep(0.05)

        progress.progress(min(1.0, i / n))

    _reset_gen_flags()

# ----------------------------
# UI
# ----------------------------
st.subheader("🎬 Generate Segment MP4s")

extract_dir = st.session_state.get("extract_dir", "")
segments = st.session_state.get("segments", [])

if not extract_dir or not Path(extract_dir).exists():
    st.warning("Upload/extract a ZIP first.")
    st.stop()

out_dir = _default_out_dir(extract_dir)
st.caption(f"Segments will be saved to: {out_dir}")

# Resolution control matches Part 1 sidebar values if you kept them;
# otherwise default to 1280x720
res = st.session_state.get("ui_resolution", "1280x720")
w, h = (1280, 720) if res == "1280x720" else (1920, 1080)

colA, colB, colC = st.columns([1, 1, 1])
with colA:
    overwrite = st.checkbox("Overwrite existing MP4s", value=False, disabled=st.session_state["is_generating"])
with colB:
    fps = st.number_input("FPS", min_value=12, max_value=60, value=30, step=1, disabled=st.session_state["is_generating"])
with colC:
    zoom_strength = st.slider(
        "Ken Burns zoom strength (zoom-only)",
        min_value=1.00,
        max_value=1.20,
        value=1.06,
        step=0.01,
        disabled=st.session_state["is_generating"],
    )

col1, col2 = st.columns([1, 1])
with col1:
    generate_clicked = st.button(
        "🚀 Generate Videos",
        type="primary",
        disabled=st.session_state["is_generating"] or (len(segments) == 0),
        use_container_width=True,
    )
with col2:
    st.button(
        "🛑 Stop after current segment",
        disabled=not st.session_state["is_generating"],
        on_click=_request_stop,
        use_container_width=True,
    )

if generate_clicked:
    _log("Starting sequential generation…")
    generate_all_segments_sequential(
        segments=segments,
        extract_dir=extract_dir,
        out_dir=out_dir,
        overwrite=overwrite,
        zoom_strength=float(zoom_strength),
        fps=int(fps),
        width=int(w),
        height=int(h),
    )

# ----------------------------
# Output previews
# ----------------------------
st.markdown("---")
st.subheader("✅ Generated MP4s")

generated = st.session_state.get("generated", {})
if not generated:
    st.info("No MP4s generated yet.")
else:
    for seg in segments:
        seg_key = seg["key"]
        mp4_path = generated.get(seg_key)
        if mp4_path and Path(mp4_path).exists():
            st.write(f"**{seg['label']}**")
            st.video(mp4_path)

st.markdown("---")
st.subheader("🧾 Log")
if st.session_state.get("gen_log"):
    st.code("\n".join(st.session_state["gen_log"][-200:]))
else:
    st.caption("Log will appear here.")

