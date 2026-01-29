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
# PART 2/2 — Generate Segment MP4s (Sequential, Crash-Safe) + DigitalOcean Spaces Export
# File: app.py
# Section: PART 2/2 (replace this whole block top-to-bottom)
# ============================

import gc
import os
import time
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import streamlit as st
import video_pipeline as vp

# boto3 (classic S3 API) — add to requirements.txt: boto3>=1.34.0
import boto3
from botocore.client import Config


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
if "spaces_upload_log" not in st.session_state:
    st.session_state["spaces_upload_log"] = []  # list[str]
if "spaces_public_urls" not in st.session_state:
    st.session_state["spaces_public_urls"] = []  # list[str]
if "spaces_last_prefix" not in st.session_state:
    st.session_state["spaces_last_prefix"] = ""  # str


def _log(msg: str) -> None:
    st.session_state["gen_log"].append(msg)


def _ulog(msg: str) -> None:
    st.session_state["spaces_upload_log"].append(msg)


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


def _get_resolution_wh() -> Tuple[int, int]:
    res = st.session_state.get("ui_resolution", "1280x720")
    return (1280, 720) if res == "1280x720" else (1920, 1080)


def _scan_mp4s(out_dir: str) -> List[str]:
    """
    Robust export: scan output directory for MP4s (works even after session_state resets).
    """
    p = Path(out_dir)
    if not p.exists():
        return []
    files = sorted([str(x) for x in p.glob("*.mp4") if x.is_file()])
    return files


def _read_spaces_secret(key: str, default: str = "") -> str:
    """
    Read from st.secrets, with env var fallback.
    Supports either:
      st.secrets["do_spaces"][...]
    or flat:
      st.secrets["DO_SPACES_KEY"], etc.
    """
    # nested
    try:
        if "do_spaces" in st.secrets and key in st.secrets["do_spaces"]:
            return str(st.secrets["do_spaces"][key]).strip()
    except Exception:
        pass

    # flat
    try:
        if key in st.secrets:
            return str(st.secrets[key]).strip()
    except Exception:
        pass

    # env fallback (optional)
    env_map = {
        "key": "DO_SPACES_KEY",
        "secret": "DO_SPACES_SECRET",
        "region": "DO_SPACES_REGION",
        "bucket": "DO_SPACES_BUCKET",
        "public_base": "DO_SPACES_PUBLIC_BASE",
        "endpoint": "DO_SPACES_ENDPOINT",  # optional override
    }
    env_name = env_map.get(key, "")
    if env_name:
        return str(os.environ.get(env_name, default)).strip()

    return default


def _spaces_client_and_context() -> Tuple[object, str, str, str]:
    """
    Returns: (s3_client, bucket, region, public_base)
    """
    key = _read_spaces_secret("key")
    secret = _read_spaces_secret("secret")
    region = _read_spaces_secret("region", "nyc3")
    bucket = _read_spaces_secret("bucket")
    public_base = _read_spaces_secret("public_base", "")

    if not key or not secret or not bucket:
        raise RuntimeError(
            "Missing DigitalOcean Spaces secrets. "
            "Set st.secrets['do_spaces']['key'/'secret'/'bucket'/'region'] "
            "and optionally 'public_base'."
        )

    endpoint_override = _read_spaces_secret("endpoint", "")
    endpoint_url = endpoint_override or f"https://{region}.digitaloceanspaces.com"

    s3 = boto3.client(
        "s3",
        region_name=region,
        endpoint_url=endpoint_url,
        aws_access_key_id=key,
        aws_secret_access_key=secret,
        config=Config(signature_version="s3v4"),
    )
    return s3, bucket, region, public_base


def _build_public_url(*, bucket: str, region: str, public_base: str, object_key: str) -> str:
    """
    If you provide DO_SPACES_PUBLIC_BASE, we use it.
    Otherwise we use the standard public URL form.
    """
    object_key = object_key.lstrip("/")
    if public_base:
        return f"{public_base.rstrip('/')}/{object_key}"
    # Standard DO Spaces public URL
    return f"https://{bucket}.{region}.digitaloceanspaces.com/{object_key}"


def _job_prefix() -> str:
    """
    Example prefix per job:
      uappress/<job_slug>/<timestamp>/
    job_slug derived from uploaded ZIP name when available; fallback "job".
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_path = st.session_state.get("zip_path", "") or ""
    zip_name = Path(zip_path).name if zip_path else ""
    base = zip_name.replace(".zip", "").strip() if zip_name else "job"
    slug = vp.safe_slug(base, max_len=50)
    return f"uappress/{slug}/{ts}/"


def _upload_file_to_spaces(
    *,
    s3,
    bucket: str,
    region: str,
    public_base: str,
    local_path: str,
    object_key: str,
    make_public: bool = True,
) -> str:
    """
    Upload local file to Spaces and return public URL.
    """
    extra = {"ContentType": "video/mp4"}
    # If your bucket is private-by-default, this can make objects public.
    # If your bucket policy already makes objects public, you can leave it on or off.
    if make_public:
        extra["ACL"] = "public-read"

    s3.upload_file(local_path, bucket, object_key, ExtraArgs=extra)
    return _build_public_url(bucket=bucket, region=region, public_base=public_base, object_key=object_key)


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

# Scene timing controls (requested)
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
        max_value=60,
        value=20,
        step=1,
        disabled=st.session_state["is_generating"],
    )
with colF:
    max_scene_seconds = st.slider(
        "Max seconds per scene",
        min_value=int(min_scene_seconds),
        max_value=120,
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
# Phase 3 — Export (DigitalOcean Spaces)
# ----------------------------
st.markdown("---")
st.subheader("☁️ Export to DigitalOcean Spaces")

mp4_paths = _scan_mp4s(out_dir)

if not mp4_paths:
    st.info("No MP4s found in the output folder yet. Generate at least one segment first.")
else:
    st.caption(f"Found **{len(mp4_paths)}** MP4(s) in `{out_dir}` (scan-based; survives reruns).")

    colL, colR = st.columns([1, 1])
    with colL:
        make_public = st.checkbox("Make uploaded files public (ACL: public-read)", value=True)
    with colR:
        prefix_override = st.text_input(
            "Spaces prefix (folder) — optional",
            value=st.session_state.get("spaces_last_prefix", ""),
            help="Leave blank to auto-generate: uappress/<job>/<timestamp>/",
        )

    upload_clicked = st.button(
        "☁️ Upload ALL MP4s to DigitalOcean Spaces",
        disabled=st.session_state["is_generating"],
        use_container_width=True,
    )

    if upload_clicked:
        st.session_state["spaces_upload_log"] = []
        st.session_state["spaces_public_urls"] = []

        try:
            s3, bucket, region, public_base = _spaces_client_and_context()

            job_prefix = (prefix_override or "").strip()
            if not job_prefix:
                job_prefix = _job_prefix()
            if not job_prefix.endswith("/"):
                job_prefix += "/"

            st.session_state["spaces_last_prefix"] = job_prefix
            _ulog(f"Bucket: {bucket} | Region: {region}")
            _ulog(f"Prefix: {job_prefix}")

            progress = st.progress(0.0)
            status = st.empty()

            for i, local_path in enumerate(mp4_paths, start=1):
                name = Path(local_path).name
                object_key = f"{job_prefix}{name}"

                status.info(f"Uploading {i}/{len(mp4_paths)} — {name}")
                try:
                    url = _upload_file_to_spaces(
                        s3=s3,
                        bucket=bucket,
                        region=region,
                        public_base=public_base,
                        local_path=local_path,
                        object_key=object_key,
                        make_public=bool(make_public),
                    )
                    st.session_state["spaces_public_urls"].append(url)
                    _ulog(f"✅ {name} -> {url}")
                except Exception as e:
                    _ulog(f"❌ {name} failed: {type(e).__name__}: {e}")

                progress.progress(min(1.0, i / max(1, len(mp4_paths))))

            status.success("Upload complete.")
        except Exception as e:
            st.error(f"Upload failed: {type(e).__name__}: {e}")
            _ulog(f"❌ Upload failed: {type(e).__name__}: {e}")

    urls = st.session_state.get("spaces_public_urls", [])
    if urls:
        st.success(f"Uploaded **{len(urls)}** file(s). Public URLs:")
        st.text_area("Public URLs (copy/paste)", value="\n".join(urls), height=180)

    if st.session_state.get("spaces_upload_log"):
        with st.expander("Upload log"):
            st.code("\n".join(st.session_state["spaces_upload_log"][-300:]))


# ----------------------------
# Output previews + per-file downloads (named files)
# ----------------------------
st.markdown("---")
st.subheader("✅ Generated MP4s")

generated = st.session_state.get("generated", {})

if not mp4_paths:
    st.info("No MP4s generated yet.")
else:
    for p in mp4_paths:
        name = Path(p).name
        st.write(f"`{name}`")
        st.video(p)

        # Optional per-file download from Streamlit (still useful even with Spaces export)
        with open(p, "rb") as f:
            st.download_button(
                label="⬇️ Download MP4",
                data=f,
                file_name=name,
                mime="video/mp4",
                key=f"dl_{name}",
            )


# ----------------------------
# Log
# ----------------------------
st.markdown("---")
st.subheader("🧾 Generation Log")
if st.session_state.get("gen_log"):
    st.code("\n".join(st.session_state["gen_log"][-200:]))
else:
    st.caption("Log will appear here.")
