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
# PART 2/2 — Generate Segment MP4s (Sequential, Crash-Safe)
# + AUTO Upload Each MP4 to DigitalOcean Spaces (no export button needed)
# File: app.py
# Section: PART 2/2 (replace this whole block top-to-bottom)
# ============================

import gc
import os
import json
import time
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import streamlit as st
import video_pipeline as vp

# boto3 (classic S3 API) — add to requirements.txt: boto3>=1.34.0
import boto3
from botocore.client import Config
from botocore.exceptions import ClientError


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

# Spaces-related
if "spaces_upload_log" not in st.session_state:
    st.session_state["spaces_upload_log"] = []  # list[str]
if "spaces_public_urls" not in st.session_state:
    st.session_state["spaces_public_urls"] = []  # list[str]
if "spaces_last_prefix" not in st.session_state:
    st.session_state["spaces_last_prefix"] = ""  # str
if "spaces_uploaded_keys" not in st.session_state:
    st.session_state["spaces_uploaded_keys"] = set()  # set[str] (in-memory for this session)
if "spaces_manifest_url" not in st.session_state:
    st.session_state["spaces_manifest_url"] = ""  # str


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
    Robust scan: find MP4s in output directory (works even after session_state resets).
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
    object_key = object_key.lstrip("/")
    if public_base:
        return f"{public_base.rstrip('/')}/{object_key}"
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


def _object_exists(*, s3, bucket: str, object_key: str) -> bool:
    try:
        s3.head_object(Bucket=bucket, Key=object_key)
        return True
    except ClientError as e:
        code = str(e.response.get("Error", {}).get("Code", ""))
        if code in ("404", "NoSuchKey", "NotFound"):
            return False
        # Any other error: re-raise
        raise


def _upload_file_to_spaces(
    *,
    s3,
    bucket: str,
    region: str,
    public_base: str,
    local_path: str,
    object_key: str,
    make_public: bool = True,
    skip_if_exists: bool = True,
) -> str:
    """
    Upload local file to Spaces and return public URL.
    Idempotent: optionally skip if object already exists.
    """
    object_key = object_key.lstrip("/")

    if skip_if_exists:
        if object_key in st.session_state.get("spaces_uploaded_keys", set()):
            return _build_public_url(bucket=bucket, region=region, public_base=public_base, object_key=object_key)
        if _object_exists(s3=s3, bucket=bucket, object_key=object_key):
            st.session_state["spaces_uploaded_keys"].add(object_key)
            return _build_public_url(bucket=bucket, region=region, public_base=public_base, object_key=object_key)

    extra = {"ContentType": "video/mp4"}
    if make_public:
        extra["ACL"] = "public-read"

    s3.upload_file(local_path, bucket, object_key, ExtraArgs=extra)
    st.session_state["spaces_uploaded_keys"].add(object_key)
    return _build_public_url(bucket=bucket, region=region, public_base=public_base, object_key=object_key)


def _upload_bytes_to_spaces(
    *,
    s3,
    bucket: str,
    region: str,
    public_base: str,
    data: bytes,
    object_key: str,
    content_type: str,
    make_public: bool = True,
) -> str:
    object_key = object_key.lstrip("/")
    extra = {"ContentType": content_type}
    if make_public:
        extra["ACL"] = "public-read"
    s3.put_object(Bucket=bucket, Key=object_key, Body=data, **extra)
    return _build_public_url(bucket=bucket, region=region, public_base=public_base, object_key=object_key)


def _write_and_upload_manifest(
    *,
    s3,
    bucket: str,
    region: str,
    public_base: str,
    job_prefix: str,
    make_public: bool,
    out_dir: str,
) -> str:
    """
    Upload a manifest after each segment so you can resume/recover if Streamlit crashes.
    Includes local filenames and any known public URLs.
    """
    mp4_paths = _scan_mp4s(out_dir)
    urls = st.session_state.get("spaces_public_urls", [])
    manifest: Dict[str, object] = {
        "job_prefix": job_prefix,
        "generated_count": len(mp4_paths),
        "generated_files": [Path(p).name for p in mp4_paths],
        "public_urls": urls,
        "updated_at": datetime.now().isoformat(),
    }
    key = f"{job_prefix}manifest.json"
    url = _upload_bytes_to_spaces(
        s3=s3,
        bucket=bucket,
        region=region,
        public_base=public_base,
        data=json.dumps(manifest, indent=2).encode("utf-8"),
        object_key=key,
        content_type="application/json",
        make_public=bool(make_public),
    )
    st.session_state["spaces_manifest_url"] = url
    return url


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
    # NEW: auto-upload knobs
    auto_upload: bool,
    make_public: bool,
    prefix_override: str,
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

    # If auto-upload enabled, init Spaces once (fast + reliable)
    s3 = bucket = region = public_base = None
    job_prefix = ""

    if auto_upload:
        try:
            s3, bucket, region, public_base = _spaces_client_and_context()
            job_prefix = (prefix_override or "").strip() or _job_prefix()
            if not job_prefix.endswith("/"):
                job_prefix += "/"
            st.session_state["spaces_last_prefix"] = job_prefix
            _ulog(f"Auto-upload enabled → Bucket: {bucket} | Region: {region}")
            _ulog(f"Prefix: {job_prefix}")
        except Exception as e:
            auto_upload = False
            _ulog(f"❌ Auto-upload disabled (failed to init Spaces): {type(e).__name__}: {e}")
            status.warning("Auto-upload disabled: could not initialize DigitalOcean Spaces. See upload log below.")

    for i, seg in enumerate(segments, start=1):
        if st.session_state.get("stop_requested"):
            _log("Stopped before starting next segment.")
            break

        seg_key = seg.get("key", f"segment_{i:02d}")
        seg_label = seg.get("label", seg_key)
        out_path = _segment_out_path(out_dir, seg)

        # If file already exists and overwrite is off, we can still auto-upload it (idempotent)
        if (not overwrite) and Path(out_path).exists():
            st.session_state["generated"][seg_key] = out_path
            status.info(f"Skipping generate (already exists): {seg_label}")
            detail.caption(f"Output: {out_path}")

            if auto_upload and s3 and bucket and region is not None:
                name = Path(out_path).name
                object_key = f"{job_prefix}{name}"
                try:
                    url = _upload_file_to_spaces(
                        s3=s3,
                        bucket=bucket,
                        region=region,
                        public_base=public_base or "",
                        local_path=out_path,
                        object_key=object_key,
                        make_public=bool(make_public),
                        skip_if_exists=True,
                    )
                    # Avoid duplicate URLs in the UI list
                    if url not in st.session_state["spaces_public_urls"]:
                        st.session_state["spaces_public_urls"].append(url)
                    _ulog(f"✅ (existing) {name} -> {url}")
                    _write_and_upload_manifest(
                        s3=s3,
                        bucket=bucket,
                        region=region,
                        public_base=public_base or "",
                        job_prefix=job_prefix,
                        make_public=bool(make_public),
                        out_dir=out_dir,
                    )
                except Exception as e:
                    _ulog(f"❌ Upload failed for existing {name}: {type(e).__name__}: {e}")

            progress.progress(min(1.0, i / n))
            continue

        status.info(f"Generating {seg_label} ({i}/{n})…")
        detail.caption(f"Output: {out_path}")

        # CRASH-SAFE: render to a temp file, then atomic rename
        tmp_path = str(Path(out_path).with_suffix(".mp4.part"))

        t0 = time.time()
        try:
            # Clean old temp file if present
            try:
                if Path(tmp_path).exists():
                    Path(tmp_path).unlink()
            except Exception:
                pass

            vp.render_segment_mp4(
                pair=seg["pair"],
                extract_dir=extract_dir,
                out_path=tmp_path,  # render to .part
                api_key=str(api_key),
                fps=int(fps),
                width=int(width),
                height=int(height),
                zoom_strength=float(zoom_strength),
                max_scenes=int(max_scenes),
                min_scene_seconds=int(min_scene_seconds),
                max_scene_seconds=int(max_scene_seconds),
            )

            # Rename only after success (prevents uploading partial files)
            Path(tmp_path).replace(out_path)

            st.session_state["generated"][seg_key] = out_path
            dt = time.time() - t0
            _log(f"✅ Generated {seg_label} in {dt:.1f}s")

            # AUTO UPLOAD RIGHT HERE (no button required)
            if auto_upload and s3 and bucket and region is not None:
                name = Path(out_path).name
                object_key = f"{job_prefix}{name}"
                status.info(f"Uploading to Spaces: {name}")
                try:
                    url = _upload_file_to_spaces(
                        s3=s3,
                        bucket=bucket,
                        region=region,
                        public_base=public_base or "",
                        local_path=out_path,
                        object_key=object_key,
                        make_public=bool(make_public),
                        skip_if_exists=True,  # idempotent (no double uploads on reruns)
                    )
                    if url not in st.session_state["spaces_public_urls"]:
                        st.session_state["spaces_public_urls"].append(url)
                    _ulog(f"✅ {name} -> {url}")

                    # Upload/update manifest after each successful segment upload
                    murl = _write_and_upload_manifest(
                        s3=s3,
                        bucket=bucket,
                        region=region,
                        public_base=public_base or "",
                        job_prefix=job_prefix,
                        make_public=bool(make_public),
                        out_dir=out_dir,
                    )
                    _ulog(f"📄 manifest.json -> {murl}")

                except Exception as e:
                    _ulog(f"❌ {name} upload failed: {type(e).__name__}: {e}")

        except Exception as e:
            _log(f"❌ Failed {seg_label}: {type(e).__name__}: {e}")
            status.error(f"Failed generating {seg_label}. See log below.")
            break
        finally:
            # Cleanup temp if still there
            try:
                if Path(tmp_path).exists():
                    Path(tmp_path).unlink()
            except Exception:
                pass
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

# Scene timing controls
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

# NEW: Auto-upload settings
st.markdown("---")
st.subheader("☁️ DigitalOcean Spaces (Auto-upload)")

colU1, colU2, colU3 = st.columns([1, 1, 1])
with colU1:
    auto_upload = st.checkbox(
        "Auto-upload each MP4 as soon as it’s created",
        value=True,
        disabled=st.session_state["is_generating"],
        help="Uploads immediately after each segment finishes. No export button required.",
    )
with colU2:
    make_public = st.checkbox(
        "Make uploaded files public (ACL: public-read)",
        value=True,
        disabled=st.session_state["is_generating"],
    )
with colU3:
    prefix_override = st.text_input(
        "Spaces prefix (folder) — optional",
        value=st.session_state.get("spaces_last_prefix", ""),
        disabled=st.session_state["is_generating"],
        help="Leave blank to auto-generate: uappress/<job>/<timestamp>/",
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
        auto_upload=bool(auto_upload),
        make_public=bool(make_public),
        prefix_override=str(prefix_override or ""),
    )


# ----------------------------
# Live outputs (URLs + Manifest)
# ----------------------------
st.markdown("---")
st.subheader("✅ Upload Results")

urls = st.session_state.get("spaces_public_urls", [])
manifest_url = st.session_state.get("spaces_manifest_url", "")

if urls:
    st.success(f"Uploaded **{len(urls)}** file(s) to Spaces:")
    st.text_area("Public URLs (copy/paste)", value="\n".join(urls), height=180)
else:
    st.caption("No uploaded URLs yet (generate a segment with auto-upload enabled).")

if manifest_url:
    st.info(f"Manifest URL: {manifest_url}")

if st.session_state.get("spaces_upload_log"):
    with st.expander("Upload log"):
        st.code("\n".join(st.session_state["spaces_upload_log"][-300:]))


# ----------------------------
# Output previews + per-file downloads (named files)
# ----------------------------
st.markdown("---")
st.subheader("🎞️ Generated MP4s (Local Preview)")

mp4_paths = _scan_mp4s(out_dir)

if not mp4_paths:
    st.info("No MP4s generated yet.")
else:
    st.caption(f"Found **{len(mp4_paths)}** MP4(s) in `{out_dir}` (scan-based; survives reruns).")
    for p in mp4_paths:
        name = Path(p).name
        st.write(f"`{name}`")
        st.video(p)

        # Optional per-file download from Streamlit
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
