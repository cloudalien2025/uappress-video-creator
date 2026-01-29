# ============================
# PART 1/2 — Sidebar + ZIP Upload + Extraction + Segment Detection (PATH-ONLY)
# ============================
# app.py — UAPpress Video Creator
#
# ✅ Sidebar exists (API key + resolution)
# ✅ ZIP saved immediately to disk (path-only in session)
# ✅ We read ZIP bytes ONLY when extracting (not stored in session)
# ✅ NO hashing, NO manifest, NO ZIP bytes in session_state
# ✅ Segments normalized for Part 3

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import List

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
    "api_key": "",         # OpenAI key (session only)
    "ui_resolution": "1280x720",
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
# Sidebar (THIS is what you were missing)
# ----------------------------
with st.sidebar:
    st.header("🔑 API Settings")
    api_key_input = st.text_input(
        "OpenAI API Key",
        type="password",
        value=st.session_state.get("api_key", ""),
        help="Required for image generation (gpt-image-1). Stored only in this session.",
    )
    st.session_state["api_key"] = (api_key_input or "").strip()

    st.divider()
    st.header("🎞️ Video Settings")
    st.session_state["ui_resolution"] = st.selectbox(
        "Resolution",
        options=["1280x720", "1920x1080"],
        index=0 if st.session_state.get("ui_resolution", "1280x720") == "1280x720" else 1,
        help="Final segment MP4 resolution.",
    )

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

        # ✅ IMPORTANT FIX:
        # video_pipeline.extract_zip_to_temp expects ZIP BYTES (not a file path).
        # So read bytes from disk (without storing them in session_state).
        with open(zip_path, "rb") as f:
            zip_bytes = f.read()

        workdir, extract_dir = vp.extract_zip_to_temp(zip_bytes)

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

st.caption("Next: Part 3 is the ONE **Generate Videos** button (sequential, crash-safe).")

# ============================
# PART 2/2 — Generate Segment MP4s (Sequential, Crash-Safe) + ZIP Export
# ============================

import gc
import time
import zipfile
from datetime import datetime
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
    st.session_state["generated"] = {}  # seg_key -> mp4_path (strings only)
if "gen_log" not in st.session_state:
    st.session_state["gen_log"] = []    # list[str]
if "zip_export_path" not in st.session_state:
    st.session_state["zip_export_path"] = ""  # path to export zip (strings only)


def _log(msg: str) -> None:
    st.session_state["gen_log"].append(msg)


def _safe_mkdir(p: str) -> str:
    Path(p).mkdir(parents=True, exist_ok=True)
    return p


def _default_out_dir(extract_dir: str) -> str:
    return _safe_mkdir(str(Path(extract_dir) / "_mp4_segments"))


def _segment_out_path(out_dir: str, seg: dict) -> str:
    """
    Option A naming:
      01_intro.mp4
      02_chapter_01_some-title.mp4
      99_outro.mp4
    Uses seg['index'], seg['key'], and optional seg['title'] slug.
    """
    idx = int(seg.get("index", 0) or 0)
    key = (seg.get("key") or f"segment_{idx:02d}").strip()

    title = (seg.get("title") or "").strip()
    slug = vp.safe_slug(title, max_len=40) if title else ""

    if slug and slug not in key:
        filename = f"{idx:02d}_{key}_{slug}.mp4"
    else:
        filename = f"{idx:02d}_{key}.mp4"

    return str(Path(out_dir) / filename)


def _request_stop() -> None:
    st.session_state["stop_requested"] = True
    _log("🛑 Stop requested — will stop after current segment finishes.")


def _reset_gen_flags() -> None:
    st.session_state["is_generating"] = False
    st.session_state["stop_requested"] = False


def _get_resolution_wh() -> tuple[int, int]:
    # Uses the sidebar selectbox from Part 1/2 if present; fallback to 1280x720.
    res = st.session_state.get("ui_resolution", "1280x720")
    return (1280, 720) if res == "1280x720" else (1920, 1080)


def _build_zip_export(*, out_dir: str, segments: list, generated: dict) -> str:
    """
    Create a ZIP containing all generated segment MP4s in the detected order.
    Returns zip file path.
    """
    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_path = str(out_dir_p / f"uappress_segments_{ts}.zip")

    files_added = 0
    missing = []

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for seg in segments:
            seg_key = seg.get("key")
            mp4_path = generated.get(seg_key)

            if not mp4_path:
                missing.append(f"{seg.get('label', seg_key)} (not generated)")
                continue

            p = Path(mp4_path)
            if not p.exists():
                missing.append(f"{seg.get('label', seg_key)} (missing file)")
                continue

            # Store in ZIP with the clean filename only
            z.write(str(p), arcname=p.name)
            files_added += 1

        # Optional: add a simple manifest
        manifest_lines = []
        manifest_lines.append("UAPpress Video Creator — Segment Export")
        manifest_lines.append(f"Created: {datetime.now().isoformat(timespec='seconds')}")
        manifest_lines.append("")
        manifest_lines.append("Segments (in order):")
        for seg in segments:
            seg_key = seg.get("key")
            mp4_path = generated.get(seg_key)
            name = Path(mp4_path).name if mp4_path else "(not generated)"
            manifest_lines.append(f"- {seg.get('index', ''):>2}  {seg.get('label', seg_key)}  ->  {name}")

        if missing:
            manifest_lines.append("")
            manifest_lines.append("Missing / Not Included:")
            for m in missing:
                manifest_lines.append(f"- {m}")

        z.writestr("manifest.txt", "\n".join(manifest_lines))

    if files_added == 0:
        # Clean up empty zip
        try:
            Path(zip_path).unlink(missing_ok=True)
        except Exception:
            pass
        raise RuntimeError("No MP4 files were available to add to the ZIP.")

    return zip_path


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
    max_scenes: int,
    min_scene_seconds: int,
    max_scene_seconds: int,
    api_key: str,
) -> None:
    st.session_state["is_generating"] = True
    st.session_state["stop_requested"] = False

    # Clear any previous ZIP export because outputs may change
    st.session_state["zip_export_path"] = ""

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
        out_path = _segment_out_path(out_dir, seg)

        # Skip if already exists unless overwrite
        if (not overwrite) and Path(out_path).exists():
            st.session_state["generated"][seg_key] = out_path
            status.info(f"Skipping (already exists): {seg_label}")
            progress.progress(min(1.0, i / n))
            continue

        status.info(f"Generating {seg_label} ({i}/{n})…")
        detail.caption(f"Output: {out_path}")

        t0 = time.time()
        try:
            vp.render_segment_mp4(
                pair=seg["pair"],
                extract_dir=extract_dir,
                out_path=out_path,
                api_key=str(api_key),
                fps=int(fps),
                width=int(width),
                height=int(height),
                zoom_strength=float(zoom_strength),
                max_scenes=int(max_scenes),
                min_scene_seconds=int(min_scene_seconds),
                max_scene_seconds=int(max_scene_seconds),
            )

            st.session_state["generated"][seg_key] = out_path
            dt = time.time() - t0
            _log(f"✅ Generated {seg_label} in {dt:.1f}s")

        except Exception as e:
            _log(f"❌ Failed {seg_label}: {type(e).__name__}: {e}")
            status.error(f"Failed generating {seg_label}. See log below.")
            break
        finally:
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

api_key = st.session_state.get("api_key", "").strip()
if not api_key:
    st.warning("Enter your OpenAI API key in the sidebar to generate videos.")
    st.stop()

out_dir = _default_out_dir(extract_dir)
st.caption(f"Segments will be saved to: {out_dir}")

w, h = _get_resolution_wh()

colA, colB, colC = st.columns([1, 1, 1])
with colA:
    overwrite = st.checkbox(
        "Overwrite existing MP4s",
        value=False,
        disabled=st.session_state["is_generating"],
    )
with colB:
    fps = st.number_input(
        "FPS",
        min_value=12,
        max_value=60,
        value=30,
        step=1,
        disabled=st.session_state["is_generating"],
    )
with colC:
    zoom_strength = st.slider(
        "Ken Burns zoom strength (zoom-only)",
        min_value=1.00,
        max_value=1.20,
        value=1.06,
        step=0.01,
        disabled=st.session_state["is_generating"],
    )

# ✅ Scene timing controls (20s min / 40s max requested)
colD, colE, colF = st.columns([1, 1, 1])
with colD:
    max_scenes = st.number_input(
        "Max scenes per segment",
        min_value=3,
        max_value=60,
        value=9,
        step=1,
        disabled=st.session_state["is_generating"],
    )
with colE:
    min_scene_seconds = st.slider(
        "Min seconds per scene",
        min_value=5,
        max_value=40,
        value=20,
        step=1,
        disabled=st.session_state["is_generating"],
    )
with colF:
    max_scene_seconds = st.slider(
        "Max seconds per scene",
        min_value=int(min_scene_seconds),
        max_value=90,
        value=40,
        step=1,
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
        max_scenes=int(max_scenes),
        min_scene_seconds=int(min_scene_seconds),
        max_scene_seconds=int(max_scene_seconds),
        api_key=api_key,
    )


# ----------------------------
# Phase 3 — Export (Downloads + ZIP)
# ----------------------------
st.markdown("---")
st.subheader("📦 Export")

generated = st.session_state.get("generated", {})

if not generated:
    st.info("Generate at least one MP4 to enable export.")
else:
    left, right = st.columns([1, 1])

    with left:
        build_zip_clicked = st.button(
            "📦 Build ZIP of all generated MP4s",
            disabled=st.session_state["is_generating"],
            use_container_width=True,
        )

    with right:
        zip_path = st.session_state.get("zip_export_path", "")
        zip_ready = bool(zip_path) and Path(zip_path).exists()
        st.caption("After building, a ZIP download button will appear here.")
        if zip_ready:
            with open(zip_path, "rb") as f:
                st.download_button(
                    label=f"⬇️ Download ZIP ({Path(zip_path).name})",
                    data=f,
                    file_name=Path(zip_path).name,
                    mime="application/zip",
                    key="dl_zip_all",
                    use_container_width=True,
                )

    if build_zip_clicked:
        try:
            zip_path = _build_zip_export(out_dir=out_dir, segments=segments, generated=generated)
            st.session_state["zip_export_path"] = zip_path
            _log(f"📦 ZIP created: {zip_path}")
            st.success(f"ZIP created: {Path(zip_path).name}")
            st.rerun()
        except Exception as e:
            _log(f"❌ ZIP build failed: {type(e).__name__}: {e}")
            st.error(f"ZIP build failed: {type(e).__name__}: {e}")


# ----------------------------
# Output previews + per-file downloads (named files)
# ----------------------------
st.markdown("---")
st.subheader("✅ Generated MP4s")

if not generated:
    st.info("No MP4s generated yet.")
else:
    for seg in segments:
        seg_key = seg.get("key")
        mp4_path = generated.get(seg_key)
        if mp4_path and Path(mp4_path).exists():
            st.write(f"**{seg.get('label', seg_key)}** — `{Path(mp4_path).name}`")
            st.video(mp4_path)

            with open(mp4_path, "rb") as f:
                st.download_button(
                    label="⬇️ Download MP4",
                    data=f,
                    file_name=Path(mp4_path).name,
                    mime="video/mp4",
                    key=f"dl_{seg_key}",
                )


st.markdown("---")
st.subheader("🧾 Log")
if st.session_state.get("gen_log"):
    st.code("\n".join(st.session_state["gen_log"][-200:]))
else:
    st.caption("Log will appear here.")
