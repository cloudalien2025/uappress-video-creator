# ============================
# app.py — UAPpress Video Creator
# Upload a TTS Studio ZIP → Generate segment MP4s (no subs, no logos, no stitching).
# + BONUS: Optional Shorts/TikTok/Reels vertical exports (UPLOAD-ONLY source MP4s)
# ============================

from __future__ import annotations

import io

import gc
import json
import os
import shutil
import tempfile
import zipfile
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import streamlit as st

try:
    import video_pipeline as vp
except Exception as e:
    st.error(f"Failed to import video_pipeline.py: {e}")
    st.stop()




# DigitalOcean Spaces (S3-compatible)
import boto3
from botocore.client import Config
from botocore.exceptions import ClientError


# ----------------------------
# Page setup
# ----------------------------
st.set_page_config(page_title="UAPpress — Video Creator", layout="wide")
st.title("🛸 UAPpress — Video Creator")
st.caption("Upload a TTS Studio ZIP → Generate segment MP4s (optional burned-in subtitles; no logos; no stitching).")




# ===============================================================
# SECTION 2 — Safe session_state init + Sidebar
# ===============================================================

DEFAULTS = {
    "api_key": "",                 # OpenAI key (session only)
    "video_mode": "Long-form (16:9)",   # output mode selector
    "ui_resolution_169": "1280x720",
    "ui_resolution_916": "1080x1920",   # vertical default
    "zip_path": "",                # absolute path to uploaded zip on disk
    "zip_root": "",                # temp dir containing the saved zip (so we can clean it)
    "workdir": "",                 # temp working directory (created by extract_zip_to_temp)
    "extract_dir": "",             # extracted zip folder
    "segments": [],                # normalized segment list
    "last_error": "",
}

for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# Generation state
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
    st.session_state["spaces_uploaded_keys"] = set()  # set[str] (session-only)
if "spaces_manifest_url" not in st.session_state:
    st.session_state["spaces_manifest_url"] = ""  # str

# BONUS shorts-maker state
if "shorts_src_choice" not in st.session_state:
    st.session_state["shorts_src_choice"] = ""
if "shorts_trim_seconds" not in st.session_state:
    st.session_state["shorts_trim_seconds"] = 60
if "shorts_make_vertical" not in st.session_state:
    st.session_state["shorts_make_vertical"] = True

# NEW: upload-only bonus source persistence
if "bonus_uploaded_mp4s" not in st.session_state:
    st.session_state["bonus_uploaded_mp4s"] = []  # list[str] (absolute paths)


# Subtitles (burn-in) state
if "burn_subtitles" not in st.session_state:
    st.session_state["burn_subtitles"] = False
if "subtitle_style" not in st.session_state:
    st.session_state["subtitle_style"] = "Auto"
if "export_srt" not in st.session_state:
    st.session_state["export_srt"] = False



with st.sidebar:
    st.header("🔑 API Settings")
    api_key_input = st.text_input(
        "OpenAI API Key",
        type="password",
        value=st.session_state.get("api_key", ""),
        help="Required for image generation + Sora. Stored only in this session.",
    )
    st.session_state["api_key"] = (api_key_input or "").strip()
    st.header("🎞️ Video Settings")

    # Mode selector (Shorts/Reels only show when selected)
    st.session_state["video_mode"] = st.selectbox(
        "Output mode",
        options=["Long-form (16:9)", "Shorts / TikTok / Reels (9:16)"],
        index=0 if st.session_state.get("video_mode", "Long-form (16:9)") == "Long-form (16:9)" else 1,
        help="Choose 16:9 for YouTube long-form, or 9:16 for Shorts/TikTok/Reels.",
        key="video_mode_select",
    )

    # Only show the relevant resolution control for the selected mode
    if st.session_state["video_mode"].startswith("Long-form"):
        st.session_state["ui_resolution_169"] = st.selectbox(
            "Resolution (16:9)",
            options=["1280x720", "1920x1080"],
            index=0 if st.session_state.get("ui_resolution_169", "1280x720") == "1280x720" else 1,
            help="Final segment MP4 resolution.",
            key="resolution_169_select",
        )
    else:
        st.session_state["ui_resolution_916"] = st.selectbox(
            "Resolution (9:16)",
            options=["720x1280", "1080x1920"],
            index=1 if st.session_state.get("ui_resolution_916", "1080x1920") == "1080x1920" else 0,
            help="Vertical resolution for Shorts/TikTok/Reels.",
            key="resolution_916_select",
        )


# ===============================================================
def extract_zip_to_temp(zip_bytes: bytes) -> tuple[str, str]:
    """Extract an uploaded ZIP into a fresh temp folder and return (workdir, extract_dir)."""
    workdir = Path(tempfile.mkdtemp(prefix="uappress_vc_"))
    extract_dir = workdir / "_extracted"
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        for member in zf.infolist():
            name = member.filename
            # Zip-slip guard
            target = (extract_dir / name).resolve()
            if not str(target).startswith(str(extract_dir.resolve())):
                continue
            zf.extract(member, extract_dir)
    return str(workdir), str(extract_dir)

# SECTION 3 — ZIP Upload + Extraction + Segment Detection (PATH-ONLY)
# ===============================================================

def _reset_zip_state() -> None:
    # Clean extract workdir
    try:
        if st.session_state.workdir and os.path.isdir(st.session_state.workdir):
            shutil.rmtree(st.session_state.workdir, ignore_errors=True)
    except Exception:
        pass

    # Clean uploaded zip root (separate temp dir)
    try:
        if st.session_state.zip_root and os.path.isdir(st.session_state.zip_root):
            shutil.rmtree(st.session_state.zip_root, ignore_errors=True)
    except Exception:
        pass

    st.session_state.zip_path = ""
    st.session_state.zip_root = ""
    st.session_state.workdir = ""
    st.session_state.extract_dir = ""
    st.session_state.segments = []
    st.session_state.last_error = ""

    # Clear bonus uploads list (paths are job-scoped under extract_dir)
    st.session_state["bonus_uploaded_mp4s"] = []
    st.session_state["shorts_src_choice"] = ""


def _save_uploaded_zip(uploaded_file) -> Tuple[str, str]:
    """
    Save uploaded ZIP to disk immediately.
    Returns (zip_root_dir, absolute_zip_path).
    """
    root = tempfile.mkdtemp(prefix="uappress_zip_")
    zip_path = os.path.join(root, "tts_studio_upload.zip")
    with open(zip_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return root, zip_path



def _safe_script_filename(p: dict) -> str:
    """Return a short script filename for UI display.

    IMPORTANT: This must never read the script contents.
    It only returns the basename of the script path (or the provided script_file).
    """
    try:
        sf = str(p.get("script_file") or "").strip()
        if sf:
            return Path(sf).name
    except Exception:
        pass
    try:
        sp = str(p.get("script_path") or "").strip()
        if sp:
            return Path(sp).name
    except Exception:
        pass
    return ""


def _normalize_segments(pairs: List[dict]) -> List[dict]:
    """
    Convert vp.pair_segments output into a stable structure expected by generator.
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

        segments.append(
            {
                "index": i,
                "key": key,
                "label": label.title(),
                "title": title,
                "script_file": Path(p.get("script_path","")).name if p.get("script_path") else "",
                "pair": p,  # original pipeline pair (audio + script paths)
            }
        )
    return segments


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

# If a new upload arrives, reset state and extract
if uploaded is not None:
    try:
        _reset_zip_state()

        zip_root, zip_path = _save_uploaded_zip(uploaded)
        st.session_state.zip_root = zip_root
        st.session_state.zip_path = zip_path

        # extract_zip_to_temp in video_pipeline accepts bytes or path.
        # Keep bytes mode here (works even if you later move zip_path).
        with open(zip_path, "rb") as f:
            zip_bytes = f.read()

        workdir, extract_dir = extract_zip_to_temp(zip_bytes)

        scripts, audios = vp.find_files(extract_dir)
        if not scripts:
            raise RuntimeError("No scripts found in ZIP.")
        if not audios:
            raise RuntimeError("No audio files found in ZIP.")

        pairs = vp.pair_segments(scripts, audios)
        segments = _normalize_segments(pairs)

        # Preflight: enforce audio pairing integrity (prevents runtime 'pair has no audio_path').
        # NOTE: `segments` are app-normalized wrappers; the actual pairing lives in seg["pair"].
        missing_audio = []
        for seg in segments:
            pair = (seg or {}).get("pair") or {}
            ap = str(pair.get("audio_path") or "").strip()
            sp = str(pair.get("script_path") or "").strip()
            # Only enforce for script-backed segments; ignore rare audio-only rows.
            if sp and not ap:
                missing_audio.append(pair)

        if missing_audio:
            labels = [
                vp.segment_label(p) + ": " + (Path(p.get("script_path") or "").name or "<no_script>")
                for p in missing_audio
            ]
            raise RuntimeError(
                "One or more segments have no matched audio file in the ZIP. "
                "Fix: ensure each script has a corresponding .mp3/.wav with the same stem, "
                "or include enough audio files for every script."
                "\n\nMissing audio for:\n- " + "\n- ".join(labels)
            )

        st.session_state.workdir = workdir
        st.session_state.extract_dir = extract_dir
        st.session_state.segments = segments
        st.session_state.last_error = ""

    except Exception as e:
        st.session_state.last_error = f"{type(e).__name__}: {e}"

if st.session_state.last_error:
    st.error(st.session_state.last_error)

if st.session_state.zip_path:
    st.success("ZIP uploaded and extracted successfully.")
    st.write(f"Detected **{len(st.session_state.segments)}** segment(s).")

    with st.expander("Show detected segments"):
        for s in st.session_state.segments:
            st.write(f"{s['index']}. **{s['label']}** — {s.get('script_file', '') or s.get('key', '')}")

st.caption("Next: Generate videos sequentially (crash-safe).")


# ===============================================================
# SECTION 4 — Shared Helpers (Logging, Paths, Resolution, Spaces)
# ===============================================================

def _log(msg: str) -> None:
    st.session_state["gen_log"].append(msg)


def _ulog(msg: str) -> None:
    st.session_state["spaces_upload_log"].append(msg)


def _safe_mkdir(p: str) -> str:
    Path(p).mkdir(parents=True, exist_ok=True)
    return p


def _default_out_dir(extract_dir: str) -> str:
    # Separate output folders by mode to avoid confusion
    mode = st.session_state.get("video_mode", "Long-form (16:9)")
    suffix = "_mp4_segments_9x16" if mode.startswith("Shorts") else "_mp4_segments"
    return _safe_mkdir(str(Path(extract_dir) / suffix))


def _brand_out_dir(extract_dir: str) -> str:
    return _safe_mkdir(str(Path(extract_dir) / "_brand_clips"))


def _shorts_out_dir(extract_dir: str) -> str:
    return _safe_mkdir(str(Path(extract_dir) / "_shorts_exports"))


def _bonus_upload_dir(extract_dir: str) -> str:
    return _safe_mkdir(str(Path(extract_dir) / "_bonus_uploads"))


def _segment_out_path(out_dir: str, seg: dict) -> str:
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
    """
    Returns (width,height) based on mode.
    IMPORTANT: Vertical output is only used when user selects Shorts/TikTok/Reels mode.
    """
    mode = st.session_state.get("video_mode", "Long-form (16:9)")
    if mode.startswith("Shorts"):
        res = st.session_state.get("ui_resolution_916", "1080x1920")
        return (720, 1280) if res == "720x1280" else (1080, 1920)
    else:
        res = st.session_state.get("ui_resolution_169", "1280x720")
        return (1280, 720) if res == "1280x720" else (1920, 1080)


def _scan_mp4s(out_dir: str) -> List[str]:
    p = Path(out_dir)
    if not p.exists():
        return []
    return sorted([str(x) for x in p.glob("*.mp4") if x.is_file()])


def _is_valid_mp4(path: str, min_bytes: int = 50_000) -> Tuple[bool, str]:
    """Basic integrity gate for uploads (prevents uploading empty/invalid artifacts)."""
    try:
        p = Path(path)
        if not p.exists():
            return False, "missing"
        if p.stat().st_size < int(min_bytes):
            return False, f"too_small<{min_bytes}B"
        # Duration check (cloud-safe: vp.ffprobe_duration_seconds falls back to ffmpeg parsing)
        try:
            dur = float(vp.ffprobe_duration_seconds(p))  # type: ignore[attr-defined]
        except Exception:
            dur = 0.0
        if dur <= 0.05:
            return False, "zero_duration"
        return True, f"ok_dur={dur:.2f}s"
    except Exception as e:
        return False, f"error:{type(e).__name__}"


def _read_spaces_secret(key: str, default: str = "") -> str:
    # st.secrets first
    try:
        if "do_spaces" in st.secrets and key in st.secrets["do_spaces"]:
            return str(st.secrets["do_spaces"][key]).strip()
    except Exception:
        pass

    # flat secrets fallback
    try:
        if key in st.secrets:
            return str(st.secrets[key]).strip()
    except Exception:
        pass

    # env fallback
    env_map = {
        "key": "DO_SPACES_KEY",
        "secret": "DO_SPACES_SECRET",
        "region": "DO_SPACES_REGION",
        "bucket": "DO_SPACES_BUCKET",
        "public_base": "DO_SPACES_PUBLIC_BASE",
        "endpoint": "DO_SPACES_ENDPOINT",
    }
    env_name = env_map.get(key, "")
    if env_name:
        return str(os.environ.get(env_name, default)).strip()

    return default


def _spaces_client_and_context():
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
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_path = st.session_state.get("zip_path", "") or ""
    zip_name = Path(zip_path).name if zip_path else ""
    base = zip_name.replace(".zip", "").strip() if zip_name else "job"
    slug = vp.safe_slug(base, max_len=50)
    # include mode in prefix for clarity
    mode = st.session_state.get("video_mode", "Long-form (16:9)")
    mode_slug = "shorts" if mode.startswith("Shorts") else "longform"
    return f"uappress/{slug}/{mode_slug}/{ts}/"


def _object_exists(*, s3, bucket: str, object_key: str) -> bool:
    try:
        s3.head_object(Bucket=bucket, Key=object_key)
        return True
    except ClientError as e:
        code = str(e.response.get("Error", {}).get("Code", ""))
        if code in ("404", "NoSuchKey", "NotFound"):
            return False
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
    content_type: str = "video/mp4",
) -> str:
    object_key = object_key.lstrip("/")

    if skip_if_exists:
        if object_key in st.session_state.get("spaces_uploaded_keys", set()):
            return _build_public_url(bucket=bucket, region=region, public_base=public_base, object_key=object_key)
        if _object_exists(s3=s3, bucket=bucket, object_key=object_key):
            st.session_state["spaces_uploaded_keys"].add(object_key)
            return _build_public_url(bucket=bucket, region=region, public_base=public_base, object_key=object_key)

    extra = {"ContentType": content_type}
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
    mp4_paths = _scan_mp4s(out_dir)
    urls = st.session_state.get("spaces_public_urls", [])
    manifest: Dict[str, object] = {
        "job_prefix": job_prefix,
        "mode": st.session_state.get("video_mode", ""),
        "resolution": f"{_get_resolution_wh()[0]}x{_get_resolution_wh()[1]}",
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
        public_base=public_base or "",
        data=json.dumps(manifest, indent=2).encode("utf-8"),
        object_key=key,
        content_type="application/json",
        make_public=bool(make_public),
    )
    st.session_state["spaces_manifest_url"] = url
    return url


# ===============================================================
# SECTION 6 — Segment MP4 Generation (Sequential, Crash-Safe) + Auto-upload
# ===============================================================

def _render_segment_mp4_compat(**kwargs) -> None:
    """
    Compatibility shim:
    - app.py expects video_pipeline.render_segment_mp4(...)
    - If pipeline exposes a different function name, try common candidates.
    """
    candidate_names = [
        "render_segment_mp4",      # expected
        "render_segment_video",
        "render_segment",
        "render_mp4_segment",
        "make_segment_mp4",
        "build_segment_mp4",
        "create_segment_mp4",
    ]

    for name in candidate_names:
        fn = getattr(vp, name, None)
        if callable(fn):
            return fn(**kwargs)

    available = [n for n in dir(vp) if ("render" in n.lower() or "segment" in n.lower() or "mp4" in n.lower())]
    raise AttributeError(
        "video_pipeline has no supported segment render function.\n"
        f"Tried: {candidate_names}\n"
        f"Found related names in video_pipeline: {available}\n"
        "Fix: either (A) add/restore render_segment_mp4 in video_pipeline.py, "
        "or (B) update app.py to call the correct function name."
    )


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
    auto_upload: bool,
    make_public: bool,
    prefix_override: str,
    burn_subtitles: bool,
    subtitle_style: str,
    export_srt: bool,
) -> list[str]:
    st.session_state["is_generating"] = True
    st.session_state["stop_requested"] = False
    outputs: list[str] = []


    progress = st.progress(0.0)
    status = st.empty()
    detail = st.empty()

    n = len(segments)
    if n == 0:
        status.error("No segments detected.")
        _reset_gen_flags()
        return []

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

        # If already exists and overwrite is off, still upload it (optional) and continue
        if (not overwrite) and Path(out_path).exists():
            st.session_state["generated"][seg_key] = out_path
            outputs.append(out_path)
            status.info(f"Skipping generate (already exists): {seg_label}")
            detail.caption(f"Output: {out_path}")

            if auto_upload and s3 and bucket and region is not None:
                name = Path(out_path).name
                object_key = f"{job_prefix}{name}"
                try:
                    ok, why = _is_valid_mp4(out_path)
                    if not ok:
                        _ulog(f"⛔ Skipping upload (invalid MP4: {why}): {name}")
                    else:
                        url = _upload_file_to_spaces(
                            s3=s3,
                            bucket=bucket,
                            region=region,
                            public_base=public_base or "",
                            local_path=out_path,
                            object_key=object_key,
                            make_public=bool(make_public),
                            skip_if_exists=True,
                            content_type="video/mp4",
                        )
                    if ok:
                        if url not in st.session_state["spaces_public_urls"]:
                            st.session_state["spaces_public_urls"].append(url)
                        _ulog(f"✅ (existing) {name} -> {url}")

                    if ok:
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
                    _ulog(f"❌ Upload failed for existing {name}: {type(e).__name__}: {e}")

            progress.progress(min(1.0, i / n))
            continue

        status.info(f"Generating {seg_label} ({i}/{n})…")
        detail.caption(f"Output: {out_path}")

        t0 = time.time()
        try:
            _render_segment_mp4_compat(
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

                burn_subtitles=bool(burn_subtitles),
                subtitle_style=str(subtitle_style or "Auto"),
                export_srt=bool(export_srt),
            )

            st.session_state["generated"][seg_key] = out_path
            outputs.append(out_path)
            dt = time.time() - t0
            _log(f"✅ Generated {seg_label} in {dt:.1f}s")

            if auto_upload and s3 and bucket and region is not None:
                name = Path(out_path).name
                object_key = f"{job_prefix}{name}"
                status.info(f"Uploading to Spaces: {name}")
                try:
                    ok, why = _is_valid_mp4(out_path)
                    if not ok:
                        _ulog(f"⛔ Skipping upload (invalid MP4: {why}): {name}")
                    else:
                        url = _upload_file_to_spaces(
                            s3=s3,
                            bucket=bucket,
                            region=region,
                            public_base=public_base or "",
                            local_path=out_path,
                            object_key=object_key,
                            make_public=bool(make_public),
                            skip_if_exists=True,
                            content_type="video/mp4",
                        )
                    if ok:
                        if url not in st.session_state["spaces_public_urls"]:
                            st.session_state["spaces_public_urls"].append(url)
                        _ulog(f"✅ {name} -> {url}")

                    if ok:
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
            gc.collect()
            time.sleep(0.05)

        progress.progress(min(1.0, i / n))

    _reset_gen_flags()

    return outputs

# ===============================================================
# SECTION 6B — BONUS: Shorts/TikTok/Reels Exporter (post-process)
# UPLOAD-ONLY SOURCE MP4s (no generated list to avoid disappearing options)
# ===============================================================

def _ffmpeg_make_vertical_clip(
    *,
    src_mp4: str,
    dst_mp4: str,
    out_w: int = 1080,
    out_h: int = 1920,
    trim_seconds: int = 60,
    keep_audio: bool = True,
) -> str:
    """
    Converts ANY source mp4 into a 9:16 mp4 using center-crop, then scales to out_w x out_h.
    Optionally trims to first N seconds (for Shorts/Reels workflows).
    Uses vp.ffmpeg_exe() + vp.run_cmd() (no new dependencies).
    """
    src = Path(src_mp4)
    dst = Path(dst_mp4)
    if not src.exists():
        raise FileNotFoundError(str(src))
    dst.parent.mkdir(parents=True, exist_ok=True)

    out_w = int(out_w)
    out_h = int(out_h)
    trim_seconds = int(trim_seconds or 0)

    # Scale to target height then center-crop width for 9:16
    vf = f"scale=-2:{out_h},crop={out_w}:{out_h}"

    ff = vp.ffmpeg_exe()

    cmd = [
        ff,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(src),
    ]

    if trim_seconds > 0:
        cmd += ["-t", str(trim_seconds)]

    cmd += [
        "-vf",
        vf,
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-pix_fmt",
        "yuv420p",
    ]

    if keep_audio:
        cmd += ["-c:a", "aac", "-b:a", os.environ.get("UAPPRESS_AAC_BITRATE", "192k")]
    else:
        cmd += ["-an"]

    cmd += ["-movflags", "+faststart", str(dst)]

    vp.run_cmd(cmd)
    return str(dst)


# ===============================================================
# SECTION 7 — UI (Generate + Branding + Results + Previews + Logs)
# ===============================================================

st.subheader("🎬 Generate Segment MP4s")


extract_dir = st.session_state.get("extract_dir", "")
segments = st.session_state.get("segments", [])
zip_ready = bool(extract_dir) and Path(extract_dir).exists()

api_key = st.session_state.get("api_key", "").strip()
api_ready = bool(api_key)

if not zip_ready:
    st.warning("Upload/extract a ZIP to generate MP4 segments. (The UI is fully available without it.)")
if not api_ready:
    st.warning("Enter your OpenAI API key in the sidebar to generate videos.")

if zip_ready:
    out_dir = _default_out_dir(extract_dir)
    st.caption(f"Segments will be saved to: {out_dir}")
else:
    out_dir = ""
    st.caption("Segments will be saved after you upload a ZIP.")

w, h = _get_resolution_wh()

# --- Crash-safe defaults for dependent sliders (Min/Max scene seconds) ---
if "min_scene_seconds" not in st.session_state:
    st.session_state["min_scene_seconds"] = 20
if "max_scene_seconds" not in st.session_state:
    st.session_state["max_scene_seconds"] = 40


def _clamp_max_scene_seconds() -> None:
    """Ensure max_scene_seconds is never below min_scene_seconds (prevents Streamlit crash)."""
    mn = int(st.session_state.get("min_scene_seconds", 20))
    mx = int(st.session_state.get("max_scene_seconds", 40))
    if mx < mn:
        st.session_state["max_scene_seconds"] = mn


colA, colB, colC = st.columns([1, 1, 1])
with colA:
    overwrite = st.checkbox(
        "Overwrite existing MP4s",
        value=False,
        disabled=st.session_state["is_generating"],
        key="overwrite_mp4s",
    )
with colB:
    fps = st.number_input(
        "FPS",
        min_value=12,
        max_value=60,
        value=30,  # YouTube-native default
        step=1,
        disabled=st.session_state["is_generating"],
        key="fps_value",
    )
with colC:
    zoom_strength = st.slider(
        "Ken Burns zoom strength (zoom-only)",
        min_value=1.00,
        max_value=1.20,
        value=1.06,
        step=0.01,
        disabled=st.session_state["is_generating"],
        key="zoom_strength_value",
    )

colD, colE, colF = st.columns([1, 1, 1])
with colD:
    max_scenes = st.number_input(
        "Max scenes per segment",
        min_value=3,
        max_value=60,
        value=9,
        step=1,
        disabled=st.session_state["is_generating"],
        key="max_scenes_value",
    )
with colE:
    min_scene_seconds = st.slider(
        "Min seconds per scene",
        min_value=5,
        max_value=60,
        step=1,
        disabled=st.session_state["is_generating"],
        key="min_scene_seconds",
        on_change=_clamp_max_scene_seconds,
    )
with colF:
    _clamp_max_scene_seconds()
    max_scene_seconds = st.slider(
        "Max seconds per scene",
        min_value=int(st.session_state["min_scene_seconds"]),
        max_value=120,
        step=1,
        disabled=st.session_state["is_generating"],
        key="max_scene_seconds",
    )


st.markdown("---")
st.subheader("📝 Subtitles (optional)")

colSub1, colSub2, colSub3 = st.columns([1, 1, 1])
with colSub1:
    burn_subtitles = st.checkbox(
        "Burn subtitles into MP4 (hardcode)",
        value=bool(st.session_state.get("burn_subtitles", False)),
        disabled=st.session_state["is_generating"],
        help="Renders captions into the video so they show everywhere (YouTube Shorts, CapCut, Instagram).",
        key="burn_subtitles",
    )
with colSub2:
    # Style options are intentionally simple + safe for Streamlit Cloud (libass).
    # Auto chooses Shorts style when in 9:16 mode, otherwise Standard.
    subtitle_style = st.selectbox(
        "Subtitle style",
        options=["Auto", "Shorts (big, center)", "Standard (bottom)"],
        index=["Auto", "Shorts (big, center)", "Standard (bottom)"].index(st.session_state.get("subtitle_style", "Auto")),
        disabled=st.session_state["is_generating"] or not bool(burn_subtitles),
        key="subtitle_style",
        help="Shorts style is larger and placed near center to avoid UI overlays.",
    )
with colSub3:
    export_srt = st.checkbox(
        "Also save .srt file next to MP4",
        value=bool(st.session_state.get("export_srt", False)),
        disabled=st.session_state["is_generating"] or not bool(burn_subtitles),
        key="export_srt",
        help="Creates a sidecar SRT file for uploads/editing. MP4 will still have burned-in subtitles.",
    )


st.markdown("---")
st.subheader("☁️ DigitalOcean Spaces (Auto-upload)")

colU1, colU2, colU3 = st.columns([1, 1, 1])
with colU1:
    auto_upload = st.checkbox(
        "Auto-upload each MP4 as soon as it’s created",
        value=True,
        disabled=st.session_state["is_generating"],
        help="Uploads immediately after each segment finishes. No export button required.",
        key="auto_upload_value",
    )
with colU2:
    make_public = st.checkbox(
        "Make uploaded files public (ACL: public-read)",
        value=True,
        disabled=st.session_state["is_generating"],
        key="make_public_value",
    )
with colU3:
    prefix_override = st.text_input(
        "Spaces prefix (folder) — optional",
        value=st.session_state.get("spaces_last_prefix", ""),
        disabled=st.session_state["is_generating"],
        help="Leave blank to auto-generate: uappress/<job>/<mode>/<timestamp>/",
        key="prefix_override_value",
    )

# --- Controls (Generate / Stop) ---
col1, col2 = st.columns([1, 1])
with col1:
    generate_clicked = st.button(
        "🚀 Generate Videos",
        type="primary",
        use_container_width=True,
        key="generate_videos_btn",
    )
with col2:
    st.button(
        "🛑 Stop after current segment",
        disabled=not st.session_state["is_generating"],
        on_click=_request_stop,
        use_container_width=True,
        key="stop_after_segment_btn",
    )

# Run generation when the button is clicked.
# Streamlit reruns top-to-bottom; the click-run is the right moment to execute.
if generate_clicked:
    if not zip_ready:
        st.warning("Upload/extract a ZIP first.")
    elif not api_ready:
        st.warning("Enter your OpenAI API key in the sidebar first.")
    elif not segments:
        st.warning("No segments detected in the extracted ZIP.")
    else:
        st.session_state["stop_requested"] = False
        st.session_state["is_generating"] = True
        try:
            zoom_strength = float(st.session_state.get("zoom_strength_value", 1.06))
            results = generate_all_segments_sequential(
                extract_dir=extract_dir,
                segments=segments,
                out_dir=out_dir,
                overwrite=overwrite,
                zoom_strength=zoom_strength,
                fps=fps,
                width=w,
                height=h,
                max_scenes=max_scenes,
                min_scene_seconds=min_scene_seconds,
                max_scene_seconds=max_scene_seconds,
                api_key=api_key,
                auto_upload=auto_upload,
                make_public=make_public,
                prefix_override=prefix_override,
                burn_subtitles=bool(st.session_state.get('burn_subtitles', False)),
                subtitle_style=str(st.session_state.get('subtitle_style', 'Auto')),
                export_srt=bool(st.session_state.get('export_srt', False)),
            )
            st.session_state["render_results"] = results or []
        finally:
            st.session_state["is_generating"] = False







# ----------------------------
# Cinematic / Visual-first Shorts — Sora Prompt Studio (Shorts mode only)
# ----------------------------
video_mode = st.session_state.get("video_mode", "Long-form (16:9)")  # sidebar selectbox key
if video_mode.startswith("Shorts"):
    st.header("🎬 Cinematic / Visual-first Shorts — Sora Prompt Studio")
    st.caption("Build a clean, brand-consistent prompt for Sora. Paste the result into Sora. This does not generate video inside the app.")

    colA, colB, colC = st.columns([1, 1, 1])
    with colA:
        sora_len_label = st.selectbox(
            "Length preset",
            ["7s (Ultra-hook)", "12s (Narrative beat)", "18s (Mini-story)"],
            index=1,
            key="sora_len_preset",
        )
    with colB:
        sora_look = st.selectbox(
            "Look",
            ["Cinematic realism", "Archival reenactment"],
            index=0,
            key="sora_look",
        )
    with colC:
        sora_camera = st.selectbox(
            "Camera motion",
            ["Slow drift (stable)", "Handheld subtle", "Orbit / parallax"],
            index=0,
            key="sora_camera",
        )

    sora_beat = st.text_area(
        "Short beat (what happens on-screen)",
        height=130,
        key="sora_short_beat",
        placeholder="Example: Night in Suffolk. A small patrol of U.S. Air Force security police moves through pine trees, red lights in the distance, fog hanging low, measured and restrained.",
    )

    sora_constraints = st.text_area(
        "Constraints / notes (optional)",
        height=90,
        key="sora_constraints",
        placeholder="Example: No gore, no monsters, no cheesy sci‑fi, no exaggerated UFO shapes, no logos. Minimal readable HUD overlays only.",
    )

    if st.button("Build Sora prompt", type="primary", key="build_sora_prompt_btn"):
        seconds = int(sora_len_label.split("s")[0])
        mode = "cinematic_realism" if sora_look == "Cinematic realism" else "archival_reenactment"
        base_prompt = (sora_beat or "").strip()
        if sora_camera:
            base_prompt += f"\nCamera: {sora_camera}."
        if sora_constraints and sora_constraints.strip():
            base_prompt += f"\nConstraints: {sora_constraints.strip()}"
        if not base_prompt.strip():
            st.warning("Add at least a short beat to generate a Sora prompt.")
        else:
            final_prompt = vp.build_sora_prompt(
                base_prompt,
                mode=mode,
                length_s=seconds,
                aspect="9:16",
                fps=30,
            )
            st.session_state["sora_prompt_built"] = final_prompt

    if st.session_state.get("sora_prompt_built"):
        st.text_area(
            "Final Sora prompt (copy/paste)",
            value=st.session_state["sora_prompt_built"],
            height=260,
            key="sora_prompt_out",
        )
        st.download_button(
            "Download prompt (.txt)",
            data=st.session_state["sora_prompt_built"].encode("utf-8"),
            file_name="uappress_sora_short_prompt.txt",
            mime="text/plain",
        )

    st.divider()

st.markdown("---")
st.subheader("3) Bonus — Shorts / TikTok / Reels Exporter (Upload-only sources)")

st.caption(
    "This is **post-processing only**. Upload any MP4 below; it will be saved into `_bonus_uploads/` "
    "for this job and will remain available across reruns. The source dropdown shows **uploads only**."
)

shorts_dir = _shorts_out_dir(extract_dir)
bonus_dir = _bonus_upload_dir(extract_dir)


def _prune_missing_bonus_paths() -> None:
    paths = st.session_state.get("bonus_uploaded_mp4s", []) or []
    kept = [p for p in paths if p and Path(p).exists()]
    st.session_state["bonus_uploaded_mp4s"] = kept


def _save_bonus_upload(uploaded_file) -> Optional[str]:
    """
    Save an uploaded MP4 into extract_dir/_bonus_uploads with collision-safe naming.
    Returns absolute path.
    """
    if uploaded_file is None:
        return None

    # Timestamp prefix prevents overwrite and makes ordering clear
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    original = (uploaded_file.name or "uploaded.mp4").strip()
    stem = Path(original).stem
    slug = vp.safe_slug(stem, max_len=60)
    filename = f"{ts}_{slug}.mp4"
    dst = Path(bonus_dir) / filename

    with open(dst, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return str(dst)


# Uploader (allow multiple)
uploaded_bonus = st.file_uploader(
    "Upload MP4(s) to use as Bonus sources",
    type=["mp4"],
    accept_multiple_files=True,
    disabled=st.session_state["is_generating"],
    key="bonus_mp4_uploader",
    help="Uploaded MP4s are saved into this job at: extract_dir/_bonus_uploads/",
)

if uploaded_bonus:
    saved_any = 0
    for uf in uploaded_bonus:
        try:
            saved_path = _save_bonus_upload(uf)
            if saved_path:
                # Add to session list (dedupe)
                cur = st.session_state.get("bonus_uploaded_mp4s", []) or []
                if saved_path not in cur:
                    cur.append(saved_path)
                    st.session_state["bonus_uploaded_mp4s"] = cur
                saved_any += 1
        except Exception as e:
            st.error(f"Bonus upload failed for {getattr(uf, 'name', 'file')}: {type(e).__name__}: {e}")

    if saved_any:
        _log(f"📥 Bonus uploads: saved {saved_any} file(s) into _bonus_uploads/")
        st.success(f"Saved {saved_any} MP4(s) into `_bonus_uploads/`.")

# Build upload-only dropdown options (scan + session, then prune)
_prune_missing_bonus_paths()
bonus_files = _scan_mp4s(bonus_dir)

# Merge scan results and session (scan-first gives stable ordering)
session_paths = st.session_state.get("bonus_uploaded_mp4s", []) or []
merged = []
for p in bonus_files + session_paths:
    if p and p not in merged and Path(p).exists():
        merged.append(p)

# Keep selection sticky: if current selection is missing, reset to blank
src_choices = [""] + merged
current_pick = st.session_state.get("shorts_src_choice", "") or ""
if current_pick and current_pick not in src_choices:
    st.session_state["shorts_src_choice"] = ""

colS1, colS2, colS3 = st.columns([1, 1, 1])
with colS1:
    src_pick = st.selectbox(
        "Source MP4 (uploads only)",
        options=src_choices,
        index=0,
        disabled=st.session_state["is_generating"],
        key="shorts_src_choice",
        help="Pick an uploaded MP4 to convert to 9:16.",
    )
with colS2:
    trim_seconds = st.number_input(
        "Trim to first N seconds (0 = no trim)",
        min_value=0,
        max_value=600,
        value=int(st.session_state.get("shorts_trim_seconds", 60)),
        step=5,
        disabled=st.session_state["is_generating"],
        key="shorts_trim_seconds",
    )
with colS3:
    vertical_res = st.selectbox(
        "Export resolution (9:16)",
        options=["720x1280", "1080x1920"],
        index=1,
        disabled=st.session_state["is_generating"],
        key="shorts_export_res",
    )

export_clicked = st.button(
    "🎬 Create 9:16 Export",
    disabled=st.session_state["is_generating"] or not bool(src_pick),
    use_container_width=True,
    key="make_vertical_export_btn",
)

if export_clicked and src_pick:
    try:
        out_w, out_h = (720, 1280) if vertical_res == "720x1280" else (1080, 1920)
        src_p = Path(src_pick)
        dst_name = src_p.stem + f"_9x16_{out_w}x{out_h}.mp4"
        dst_path = str(Path(shorts_dir) / dst_name)

        st.info("Creating 9:16 export via ffmpeg…")
        _log(f"🎬 Bonus export: {src_p.name} -> {dst_name} (trim={int(trim_seconds)}s)")

        _ffmpeg_make_vertical_clip(
            src_mp4=str(src_p),
            dst_mp4=dst_path,
            out_w=out_w,
            out_h=out_h,
            trim_seconds=int(trim_seconds),
            keep_audio=True,
        )

        st.success(f"Created: {dst_path}")
        st.video(dst_path)

        with open(dst_path, "rb") as f:
            st.download_button(
                label="⬇️ Download 9:16 MP4",
                data=f,
                file_name=Path(dst_path).name,
                mime="video/mp4",
                key=f"dl_bonus_{Path(dst_path).name}",
            )

    except Exception as e:
        st.error(f"Bonus export failed: {type(e).__name__}: {e}")
        _log(f"❌ Bonus export failed: {type(e).__name__}: {e}")

# Optional auto-upload for bonus exports
with st.expander("Auto-upload bonus exports to Spaces (optional)", expanded=False):
    st.caption("This uploads files from `_shorts_exports/` to your Spaces prefix.")
    bonus_upload = st.button(
        "☁️ Upload ALL bonus exports now",
        disabled=st.session_state["is_generating"],
        use_container_width=True,
        key="upload_bonus_exports_btn",
    )
    if bonus_upload:
        try:
            s3, bucket, region, public_base = _spaces_client_and_context()
            job_prefix = (st.session_state.get("spaces_last_prefix", "") or _job_prefix()).strip()
            if not job_prefix.endswith("/"):
                job_prefix += "/"
            bonus_prefix = job_prefix + "shorts_exports/"
            _ulog(f"Bonus upload → {bonus_prefix}")

            bonus_files = _scan_mp4s(shorts_dir)
            if not bonus_files:
                st.info("No bonus exports found yet.")
            else:
                for p in bonus_files:
                    name = Path(p).name
                    object_key = f"{bonus_prefix}{name}"
                    url = _upload_file_to_spaces(
                        s3=s3,
                        bucket=bucket,
                        region=region,
                        public_base=public_base or "",
                        local_path=p,
                        object_key=object_key,
                        make_public=True,
                        skip_if_exists=False,
                        content_type="video/mp4",
                    )
                    if url not in st.session_state["spaces_public_urls"]:
                        st.session_state["spaces_public_urls"].append(url)
                    _ulog(f"✅ bonus {name} -> {url}")

                st.success(f"Uploaded {len(bonus_files)} bonus export(s).")
        except Exception as e:
            st.error(f"Bonus upload failed: {type(e).__name__}: {e}")
            _ulog(f"❌ Bonus upload failed: {type(e).__name__}: {e}")

st.markdown("---")
st.subheader("🧾 Generation Log")
if st.session_state.get("gen_log"):
    st.code("\n".join(st.session_state["gen_log"][-200:]))
else:
    st.caption("Log will appear here.")
