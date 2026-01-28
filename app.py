# ============================
# PART 1/5 — Core Setup (NO ZIP HASH) + Sidebar + Safe Session State + ZIP saved to disk
# ============================
# app.py — UAPpress Video Creator (TTS ZIP → Generate segment MP4s)
#
# ✅ NO ZIP hash logic (deleted)
# ✅ Never store big ZIP bytes in session_state
# ✅ Upload ZIP → immediately save to a temp file on disk
# ✅ session_state stores only small strings (paths), avoiding Streamlit crashes
# ✅ Safe session_state init (no AttributeError ever)

from __future__ import annotations

import os
import io
import re
import json
import time
import zipfile
import shutil
import tempfile
from typing import Dict, List, Optional, Tuple

import streamlit as st
from openai import OpenAI
import imageio_ffmpeg

import video_pipeline as vp  # <-- your shared helpers live in video_pipeline.py

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()

# ----------------------------
# Page setup
# ----------------------------
st.set_page_config(page_title="UAPpress — Video Creator", layout="wide")
st.title("🛸 UAPpress — Video Creator (TTS ZIP → Segment MP4s)")
st.caption("Upload a TTS Studio ZIP (scripts + audio). Generate segment MP4s. (No subs, no logos, no final stitch.)")

# ----------------------------
# Safe session_state init
# ----------------------------
DEFAULTS = {
    "api_key": "",
    "zip_local_path": "",     # where the uploaded zip is saved on disk
    "workdir": "",            # temp working folder (optional)
    "extract_dir": "",        # extracted zip folder (optional)
    "segments": [],           # list[dict] pairs from vp.pair_segments
    "zip_uploaded_name": "",  # UI only
    "last_error": "",
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ----------------------------
# Sidebar
# ----------------------------
with st.sidebar:
    st.header("🔑 OpenAI API Key")
    st.session_state.api_key = st.text_input(
        "Paste your OpenAI API key",
        value=st.session_state.api_key,
        type="password",
        help="This is not saved anywhere except this session.",
    )

    st.divider()
    st.header("📦 Output")
    resolution = st.selectbox(
        "Resolution",
        options=["1280x720", "1920x1080"],
        index=0,
        help="Applied to generated scene clips (Ken Burns).",
        key="ui_resolution",
    )

    cache_dir = st.text_input(
        "Cache directory",
        value=os.environ.get("UAPPRESS_CACHE_DIR", ".uappress_cache"),
        help="Disk cache for generated images/clips. Keeping it helps cost/speed.",
        key="ui_cache_dir",
    )
    os.environ["UAPPRESS_CACHE_DIR"] = cache_dir

    if st.button("🧹 Clear cache", use_container_width=True):
        try:
            if os.path.isdir(cache_dir):
                shutil.rmtree(cache_dir, ignore_errors=True)
            os.makedirs(cache_dir, exist_ok=True)
            st.success("Cache cleared.")
        except Exception as e:
            st.error(f"Could not clear cache: {e}")

# ----------------------------
# Helpers (ZIP persistence)
# ----------------------------
def _save_uploaded_zip_to_disk(uploaded_file) -> str:
    """
    Save uploaded ZIP to a stable temp folder.
    Returns the full path. Stores NO bytes in session_state.
    """
    if uploaded_file is None:
        return ""

    root = tempfile.mkdtemp(prefix="uappress_zip_")
    zip_path = os.path.join(root, "tts_studio_upload.zip")

    # Streamlit UploadedFile supports getbuffer()
    with open(zip_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return zip_path

def _reset_zip_state() -> None:
    # Cleanup old workdir if present
    try:
        if st.session_state.workdir and os.path.isdir(st.session_state.workdir):
            shutil.rmtree(st.session_state.workdir, ignore_errors=True)
    except Exception:
        pass

    st.session_state.zip_local_path = ""
    st.session_state.workdir = ""
    st.session_state.extract_dir = ""
    st.session_state.segments = []
    st.session_state.zip_uploaded_name = ""
    st.session_state.last_error = ""

# ----------------------------
# Upload ZIP
# ----------------------------
st.subheader("1) Upload ZIP from TTS Studio (scripts + audio)")
uploaded = st.file_uploader(
    "TTS Studio ZIP",
    type=["zip"],
    accept_multiple_files=False,
    help="Must contain scripts (txt/md/json) and audio (mp3/wav/m4a...).",
)

colA, colB = st.columns([1, 1])

with colA:
    if st.button("❌ Remove ZIP", disabled=(uploaded is None and not st.session_state.zip_local_path), use_container_width=True):
        _reset_zip_state()
        st.rerun()

# If user uploads a ZIP, immediately save it to disk and extract+detect segments.
if uploaded is not None:
    try:
        # If this is a new file name, reset previous state first
        if st.session_state.zip_uploaded_name and st.session_state.zip_uploaded_name != uploaded.name:
            _reset_zip_state()

        st.session_state.zip_uploaded_name = uploaded.name
        st.session_state.zip_local_path = _save_uploaded_zip_to_disk(uploaded)

        # Extract + discover + pair (this is cheap; no OpenAI calls)
        with open(st.session_state.zip_local_path, "rb") as f:
            zip_bytes = f.read()

        workdir, extract_dir = vp.extract_zip_to_temp(zip_bytes)
        scripts, audios = vp.find_files(extract_dir)
        pairs = vp.pair_segments(scripts, audios)

        st.session_state.workdir = workdir
        st.session_state.extract_dir = extract_dir
        st.session_state.segments = pairs
        st.session_state.last_error = ""

    except Exception as e:
        st.session_state.last_error = f"{type(e).__name__}: {e}"

# Status + segment preview
if st.session_state.last_error:
    st.error(st.session_state.last_error)

if st.session_state.zip_local_path:
    st.success(f"ZIP saved to disk: {st.session_state.zip_uploaded_name}")
    st.write(f"Detected **{len(st.session_state.segments)}** segments.")
    if st.session_state.segments:
        with st.expander("Show detected segments"):
            for i, p in enumerate(st.session_state.segments, start=1):
                st.write(f"{i}. **{vp.segment_label(p)}** — {p.get('title_guess','')}")

# NOTE:
# Part 2 will add scene timing controls + the OpenAI client initializer.
# Part 3 will be the ONE button "Generate Videos" sequential loop (no subs, no logos).

# ============================
# PART 2/5 — ZIP Upload + Extraction + Pairing (Path-only, No Manifest)
# ============================

import os
import time
import tempfile
from pathlib import Path
import streamlit as st

# Assumes you already have these from video_pipeline.py:
# - extract_zip_to_temp(zip_path_or_bytes, ...)  (we will call a file-path version)
# - find_files(extract_dir)
# - pair_segments(scripts, audios)
# - segment_label(p)
# - safe_slug(text)
# - read_script_file(path)  (optional preview)

def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def _save_uploaded_zip_to_disk(uploaded_file, base_dir: str) -> str:
    """
    Save uploaded ZIP immediately; return absolute file path.
    Uses timestamped filename to avoid collisions without hashing.
    """
    _ensure_dir(base_dir)
    ts = time.strftime("%Y%m%d_%H%M%S")
    zip_path = os.path.join(base_dir, f"tts_studio_{ts}.zip")
    with open(zip_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return zip_path

def _build_segments(pairs: list[dict]) -> list[dict]:
    """
    Convert pair dicts to a stable segment structure used by Part 3.
    Enforce order: Intro -> Chapter 1..N -> Outro -> others
    """
    def label_of(p): return segment_label(p)  # expected "INTRO", "CHAPTER 1", "OUTRO", etc.

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
    for idx, p in enumerate(ordered, start=1):
        label = label_of(p)  # e.g. "INTRO" / "CHAPTER 1" / "OUTRO"
        title = p.get("title_guess") or ""
        # stable key for filenames
        key = safe_slug(label.lower().replace(" ", "_"))
        if key.startswith("chapter_") and p.get("chapter_no") is not None:
            key = f"chapter_{int(p['chapter_no']):02d}"

        segments.append({
            "index": idx,
            "key": key,
            "label": label.title() if label.isupper() else label,
            "title": title,
            # keep original pipeline pair for rendering
            "pair": p,
        })
    return segments

# ----------------------------
# 1) Upload ZIP
# ----------------------------
st.subheader("1) Upload ZIP from TTS Studio (scripts + audio)")
zip_file = st.file_uploader("TTS Studio ZIP", type=["zip"], key="tts_zip_video_creator")

# Base workspace where we store uploaded zips + extracted folders
# (Could be a cache dir you already use; keep it path-based)
WORK_BASE = os.environ.get("UAPPRESS_WORK_DIR", str(Path(tempfile.gettempdir()) / "uappress_video_creator"))
_ensure_dir(WORK_BASE)

# Reset on new upload
if zip_file:
    zip_path = _save_uploaded_zip_to_disk(zip_file, base_dir=WORK_BASE)

    st.session_state["zip_path"] = zip_path
    st.session_state["extract_dir"] = ""
    st.session_state["segments"] = []
    st.session_state["generated"] = {}     # segment_key -> mp4_path
    st.session_state["gen_log"] = []

# If no zip_path, stop early
if not st.session_state.get("zip_path"):
    st.info("Upload the TTS Studio ZIP to detect segments.")
    st.stop()

zip_path = st.session_state["zip_path"]

# ----------------------------
# 2) Extract + pair segments (ONLY if not done yet)
# ----------------------------
if not st.session_state.get("segments"):
    with st.spinner("Extracting ZIP and detecting segments…"):
        # IMPORTANT: this must accept a FILE PATH, not zip bytes.
        # If your current extract_zip_to_temp expects bytes, we should update it
        # or add a new helper in video_pipeline.py like extract_zip_path_to_temp(zip_path).
        workdir, extract_dir = extract_zip_to_temp(zip_path)  # <-- must support zip_path

        scripts, audios = find_files(extract_dir)
        if not scripts:
            st.error("No scripts found in ZIP. Expected scripts/*.txt")
            st.stop()
        if not audios:
            st.error("No audio found in ZIP. Expected audio/*.mp3 (or wav)")
            st.stop()

        pairs = pair_segments(scripts, audios)
        segments = _build_segments(pairs)

        st.session_state["extract_dir"] = extract_dir
        st.session_state["segments"] = segments

    st.success(f"Detected {len(st.session_state['segments'])} segment(s).")

# ----------------------------
# 3) Preview detected order
# ----------------------------
st.markdown("### Detected generation order")
for s in st.session_state["segments"]:
    label = s["label"]
    title = s.get("title") or "Untitled"
    st.write(f"{s['index']}. [{label}] {title}")

st.caption("Next: Part 3 adds the ONE button **Generate Videos** + sequential loop.")


# ============================================================
# Part 3 — Generate Videos (sequential, stable, no crashes)
# ============================================================

import gc
import time
from pathlib import Path
import streamlit as st

# ---- Expected session_state from Part 1/2 (already set earlier) ----
# st.session_state["zip_path"]        -> str (path to uploaded zip saved on disk)
# st.session_state["extract_dir"]     -> str (path to extracted zip folder)
# st.session_state["segments"]        -> list[dict] (detected segments in order)
#   Each segment dict should have at least:
#     - "key"   (e.g., "intro", "chapter_01", "outro")
#     - "label" (e.g., "Intro", "Chapter 1", "Outro")

# ---- State for generation ----
if "is_generating" not in st.session_state:
    st.session_state["is_generating"] = False
if "generated" not in st.session_state:
    st.session_state["generated"] = {}  # segment_key -> mp4_path (strings only)
if "gen_log" not in st.session_state:
    st.session_state["gen_log"] = []    # list[str]
if "stop_requested" not in st.session_state:
    st.session_state["stop_requested"] = False


def _log(msg: str) -> None:
    st.session_state["gen_log"].append(msg)


def _safe_mkdir(p: str) -> str:
    Path(p).mkdir(parents=True, exist_ok=True)
    return p


def _default_out_dir(extract_dir: str) -> str:
    # Keep outputs inside extracted workspace so everything stays local + path-based
    return _safe_mkdir(str(Path(extract_dir) / "_mp4_segments"))


def _segment_out_path(out_dir: str, segment_key: str) -> str:
    # Stable filenames so resume/re-run is predictable
    return str(Path(out_dir) / f"{segment_key}.mp4")


def _reset_generation_state() -> None:
    st.session_state["is_generating"] = False
    st.session_state["stop_requested"] = False


def _request_stop() -> None:
    st.session_state["stop_requested"] = True
    _log("Stop requested — will stop after current segment finishes.")


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
    """
    Generates one MP4 at a time in a single run.
    Avoids storing large objects in session_state.
    Updates Streamlit UI incrementally to reduce crash risk.
    """
    st.session_state["is_generating"] = True
    st.session_state["stop_requested"] = False

    progress = st.progress(0.0)
    status = st.empty()
    detail = st.empty()

    n = len(segments)
    if n == 0:
        status.error("No segments detected.")
        _reset_generation_state()
        return

    for i, seg in enumerate(segments):
        if st.session_state.get("stop_requested"):
            _log("Stopped before starting next segment.")
            break

        seg_key = seg.get("key") or f"segment_{i+1:02d}"
        seg_label = seg.get("label") or seg_key
        out_path = _segment_out_path(out_dir, seg_key)

        # Skip if already generated (unless overwrite)
        if (not overwrite) and Path(out_path).exists():
            st.session_state["generated"][seg_key] = out_path
            status.info(f"Skipping (already exists): {seg_label}")
            progress.progress(min(1.0, (i + 1) / n))
            continue

        status.info(f"Generating {seg_label} ({i+1}/{n})…")
        detail.caption(f"Output: {out_path}")

        t0 = time.time()
        try:
            # ============================================================
            # ✅ THE ONLY LINE YOU MAY NEED TO CHANGE:
            # Call your video_pipeline function that renders ONE segment MP4.
            #
            # Example expected signature (adapt as needed):
            #   render_segment_mp4(
            #       segment=seg,
            #       extract_dir=extract_dir,
            #       out_path=out_path,
            #       ken_burns=True,
            #       zoom_only=True,
            #       zoom_strength=zoom_strength,
            #       fps=fps,
            #       width=width,
            #       height=height,
            #   )
            #
            # Replace `render_segment_mp4` with your actual function name.
            # ============================================================

            render_segment_mp4(   # <-- CHANGE THIS NAME IF YOUR PIPELINE CALL DIFFERS
                segment=seg,
                extract_dir=extract_dir,
                out_path=out_path,
                ken_burns=True,
                zoom_only=True,
                zoom_strength=zoom_strength,
                fps=fps,
                width=width,
                height=height,
            )

            # Persist only the path (string)
            st.session_state["generated"][seg_key] = out_path

            dt = time.time() - t0
            _log(f"✅ Generated {seg_label} in {dt:.1f}s")

        except Exception as e:
            _log(f"❌ Failed {seg_label}: {type(e).__name__}: {e}")
            status.error(f"Failed generating {seg_label}. See log below.")
            # Stop on first failure to keep state clean + avoid cascade crashes
            break
        finally:
            # Aggressively free memory between segments (helps Streamlit stability)
            gc.collect()
            time.sleep(0.05)

        progress.progress(min(1.0, (i + 1) / n))

    _reset_generation_state()

# ----------------------------
# UI controls + single button
# ----------------------------

st.subheader("🎬 Generate Segment MP4s")

extract_dir = st.session_state.get("extract_dir", "")
segments = st.session_state.get("segments", [])

colA, colB, colC = st.columns([1, 1, 1])
with colA:
    overwrite = st.checkbox("Overwrite existing MP4s", value=False, disabled=st.session_state["is_generating"])
with colB:
    fps = st.number_input("FPS", min_value=12, max_value=60, value=30, step=1, disabled=st.session_state["is_generating"])
with colC:
    zoom_strength = st.slider("Ken Burns zoom strength (zoom-only)", min_value=1.00, max_value=1.20, value=1.06, step=0.01, disabled=st.session_state["is_generating"])

colD, colE = st.columns([1, 1])
with colD:
    width = st.number_input("Width", min_value=640, max_value=3840, value=1920, step=10, disabled=st.session_state["is_generating"])
with colE:
    height = st.number_input("Height", min_value=360, max_value=2160, value=1080, step=10, disabled=st.session_state["is_generating"])

if not extract_dir or not Path(extract_dir).exists():
    st.warning("Upload/extract a ZIP first (Part 1/2).")
else:
    out_dir = _default_out_dir(extract_dir)
    st.caption(f"Segments will be saved to: {out_dir}")

    col1, col2 = st.columns([1, 1])
    with col1:
        generate_clicked = st.button(
            "🚀 Generate Videos",
            type="primary",
            disabled=st.session_state["is_generating"] or (len(segments) == 0),
        )
    with col2:
        st.button(
            "🛑 Stop after current segment",
            disabled=not st.session_state["is_generating"],
            on_click=_request_stop,
        )

    if generate_clicked:
        # Clear stop flag + run sequential generation in this single execution
        st.session_state["stop_requested"] = False
        _log("Starting sequential generation…")
        generate_all_segments_sequential(
            segments=segments,
            extract_dir=extract_dir,
            out_dir=out_dir,
            overwrite=overwrite,
            zoom_strength=float(zoom_strength),
            fps=int(fps),
            width=int(width),
            height=int(height),
        )

# ----------------------------
# Output section
# ----------------------------
st.markdown("---")
st.subheader("✅ Generated MP4s")

generated = st.session_state.get("generated", {})
if not generated:
    st.info("No MP4s generated yet.")
else:
    # Display in the same order as segments
    for seg in segments:
        seg_key = seg.get("key")
        seg_label = seg.get("label", seg_key)
        mp4_path = generated.get(seg_key)
        if mp4_path and Path(mp4_path).exists():
            st.write(f"**{seg_label}**")
            st.video(mp4_path)

st.markdown("---")
st.subheader("🧾 Log")
if st.session_state.get("gen_log"):
    st.code("\n".join(st.session_state["gen_log"][-200:]))
else:
    st.caption("Log will appear here.")

# ============================
# PART 4/5 — Local Artifact Browser + Troubleshooting Tools + Guardrails
# ============================

st.divider()
st.subheader("4) Local Artifacts (For Troubleshooting)")

st.caption(
    "If something fails, the per-segment work folders (when present) can help you debug. "
    "Successful segments usually have their workdir cleared after upload to save disk."
)

work_root = os.path.join(_manifest_dir(cache_dir), "work")
_ensure_dir(work_root)

# List segment workdirs that still exist
existing_workdirs = []
try:
    for name in sorted(os.listdir(work_root)):
        p = os.path.join(work_root, name)
        if os.path.isdir(p):
            existing_workdirs.append(p)
except Exception:
    existing_workdirs = []

if not existing_workdirs:
    st.info("No per-segment work folders found (this is normal if uploads succeeded and cleanup ran).")
else:
    pick = st.selectbox(
        "Select a segment work folder",
        options=existing_workdirs,
        format_func=lambda p: os.path.basename(p),
    )

    # Show files inside
    files = []
    for root, _, fns in os.walk(pick):
        for fn in fns:
            files.append(os.path.join(root, fn))
    files = sorted(files)

    st.write(f"Files in `{pick}`:")
    for fp in files[:120]:
        st.write("-", os.path.relpath(fp, pick))
    if len(files) > 120:
        st.caption(f"...and {len(files) - 120} more")

    # Quick preview buttons for common artifacts
    preview_mp4s = [f for f in files if f.lower().endswith(".mp4")]
    preview_jsons = [f for f in files if f.lower().endswith(".json")]
    preview_txts = [f for f in files if f.lower().endswith((".txt", ".md", ".log"))]

    col1, col2, col3 = st.columns(3)

    with col1:
        if preview_mp4s:
            mp4_pick = st.selectbox("Preview MP4", preview_mp4s, key="prev_mp4")
            if st.button("▶️ Show MP4"):
                st.video(mp4_pick)
        else:
            st.caption("No MP4s in this workdir.")

    with col2:
        if preview_jsons:
            js_pick = st.selectbox("View JSON", preview_jsons, key="prev_json")
            if st.button("🔎 Show JSON"):
                try:
                    with open(js_pick, "r", encoding="utf-8") as f:
                        st.code(f.read()[:20000], language="json")
                except Exception as e:
                    st.error(f"Could not read JSON: {e}")
        else:
            st.caption("No JSON files in this workdir.")

    with col3:
        if preview_txts:
            tx_pick = st.selectbox("View text/log", preview_txts, key="prev_txt")
            if st.button("🧾 Show text/log"):
                try:
                    with open(tx_pick, "r", encoding="utf-8", errors="ignore") as f:
                        st.code(f.read()[-20000:], language="text")
                except Exception as e:
                    st.error(f"Could not read file: {e}")
        else:
            st.caption("No text/log files in this workdir.")


# ----------------------------
# Guardrails / reminders
# ----------------------------
st.divider()
st.subheader("5) Guardrails")

st.markdown(
    """
- **Subtitles are NOT burned in** (by design). You'll handle captions in CapCut.
- **OpenAI API key** is entered in the sidebar and stored only in session memory.
- **Spaces credentials** should be stored in Streamlit Secrets or environment variables — never in GitHub.
- If a segment fails:
  1) open the **Status Dashboard** (Part 4),
  2) inspect its error,
  3) optionally inspect remaining workdir artifacts here,
  4) fix the underlying issue,
  5) **Mark PENDING** and rerun **Generate Videos**.
"""
)

