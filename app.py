# app_py_GODMODE.txt
# Save this file as: app.py
#
# GODMODE Streamlit Orchestrator (thin shell)
# ZIP upload (MP3 + scripts) → deterministic segment render → optional DigitalOcean Spaces auto-upload
#
# Non-negotiables respected:
# - ZIP-based input
# - OpenAI as AI backbone (images)
# - DigitalOcean Spaces output
# - Streamlit + GitHub compatible
#
# Design goals:
# - Fast iteration velocity
# - Rerun-safe state
# - Cost disciplined (no unnecessary API calls)
# - Minimal UI (Ferrari: clean, purposeful)

from __future__ import annotations

import os
import json
import time
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import streamlit as st

try:
    import video_pipeline as vp
except Exception as e:
    st.error(f"Failed to import video_pipeline.py: {type(e).__name__}: {e}")
    st.stop()

# DigitalOcean Spaces (S3-compatible)
import boto3
from botocore.client import Config
from botocore.exceptions import ClientError


# ----------------------------
# Page
# ----------------------------
st.set_page_config(page_title="UAPpress — GODMODE", layout="wide")
st.title("⚡ UAPpress — GODMODE")
st.caption("ZIP in → MP4s out → Spaces. Fast. Deterministic. Cheap.")

# ----------------------------
# Session state (rerun-safe)
# ----------------------------
DEFAULTS = {
    "api_key": "",
    "mode": "Long-form (16:9)",
    "res_169": "1280x720",
    "res_916": "1080x1920",
    "zip_root": "",
    "zip_path": "",
    "workdir": "",
    "extract_dir": "",
    "pairs": [],
    "generated": {},  # idx_str -> mp4_path
    "gen_log": [],
    "is_generating": False,
    "stop_requested": False,
    # Spaces
    "spaces_upload_log": [],
    "spaces_public_urls": [],
    "spaces_last_prefix": "",
    "spaces_uploaded_keys": set(),  # session-only
    "spaces_manifest_url": "",
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


def _log(msg: str) -> None:
    st.session_state["gen_log"].append(msg)


def _ulog(msg: str) -> None:
    st.session_state["spaces_upload_log"].append(msg)


def _reset_all() -> None:
    # Clean extracted workdir
    try:
        wd = st.session_state.get("workdir", "")
        if wd and os.path.isdir(wd):
            shutil.rmtree(wd, ignore_errors=True)
    except Exception:
        pass

    # Clean uploaded zip root
    try:
        zr = st.session_state.get("zip_root", "")
        if zr and os.path.isdir(zr):
            shutil.rmtree(zr, ignore_errors=True)
    except Exception:
        pass

    # Reset fields
    for k in ("zip_root", "zip_path", "workdir", "extract_dir"):
        st.session_state[k] = ""
    st.session_state["pairs"] = []
    st.session_state["generated"] = {}
    st.session_state["gen_log"] = []
    st.session_state["is_generating"] = False
    st.session_state["stop_requested"] = False

    st.session_state["spaces_upload_log"] = []
    st.session_state["spaces_public_urls"] = []
    st.session_state["spaces_last_prefix"] = ""
    st.session_state["spaces_uploaded_keys"] = set()
    st.session_state["spaces_manifest_url"] = ""


def _save_uploaded_zip(uploaded_file) -> Tuple[str, str]:
    root = tempfile.mkdtemp(prefix="uappress_zip_")
    zip_path = os.path.join(root, "tts_studio_upload.zip")
    with open(zip_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return root, zip_path


def _get_resolution_wh() -> Tuple[int, int]:
    mode = st.session_state.get("mode", "Long-form (16:9)")
    if mode.startswith("Shorts"):
        res = st.session_state.get("res_916", "1080x1920")
        return (720, 1280) if res == "720x1280" else (1080, 1920)
    res = st.session_state.get("res_169", "1280x720")
    return (1280, 720) if res == "1280x720" else (1920, 1080)


def _safe_mkdir(p: str) -> str:
    Path(p).mkdir(parents=True, exist_ok=True)
    return p


def _out_dir(extract_dir: str) -> str:
    w, h = _get_resolution_wh()
    suffix = "_mp4_segments_9x16" if h > w else "_mp4_segments"
    return _safe_mkdir(str(Path(extract_dir) / suffix))


def _segment_out_path(out_dir: str, idx: int, pair: dict) -> str:
    label = vp.segment_label(pair)
    title = (pair.get("title_guess") or "").strip()
    stem = vp.safe_slug(title or label, max_len=44)
    return str(Path(out_dir) / f"{idx:02d}_{stem}.mp4")


# ----------------------------
# DigitalOcean Spaces helpers
# ----------------------------
def _read_spaces_secret(key: str, default: str = "") -> str:
    try:
        if "do_spaces" in st.secrets and key in st.secrets["do_spaces"]:
            return str(st.secrets["do_spaces"][key]).strip()
    except Exception:
        pass
    try:
        if key in st.secrets:
            return str(st.secrets[key]).strip()
    except Exception:
        pass

    env_map = {
        "key": "DO_SPACES_KEY",
        "secret": "DO_SPACES_SECRET",
        "region": "DO_SPACES_REGION",
        "bucket": "DO_SPACES_BUCKET",
        "public_base": "DO_SPACES_PUBLIC_BASE",
        "endpoint": "DO_SPACES_ENDPOINT",
    }
    env = env_map.get(key, "")
    if env:
        return str(os.environ.get(env, default)).strip()
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
            "Set st.secrets['do_spaces']['key'/'secret'/'bucket'/'region'] (optional public_base)."
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


def _build_public_url(bucket: str, region: str, public_base: str, object_key: str) -> str:
    object_key = object_key.lstrip("/")
    if public_base:
        return f"{public_base.rstrip('/')}/{object_key}"
    return f"https://{bucket}.{region}.digitaloceanspaces.com/{object_key}"


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
            return _build_public_url(bucket, region, public_base, object_key)
        if _object_exists(s3=s3, bucket=bucket, object_key=object_key):
            st.session_state["spaces_uploaded_keys"].add(object_key)
            return _build_public_url(bucket, region, public_base, object_key)

    extra = {"ContentType": content_type}
    if make_public:
        extra["ACL"] = "public-read"
    s3.upload_file(local_path, bucket, object_key, ExtraArgs=extra)
    st.session_state["spaces_uploaded_keys"].add(object_key)
    return _build_public_url(bucket, region, public_base, object_key)


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
    return _build_public_url(bucket, region, public_base, object_key)


def _job_prefix() -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode_slug = "shorts" if st.session_state.get("mode", "").startswith("Shorts") else "longform"
    return f"uappress/godmode/{mode_slug}/{ts}/"


def _is_valid_mp4(path: str, min_bytes: int = 80_000) -> Tuple[bool, str]:
    try:
        p = Path(path)
        if not p.exists():
            return False, "missing"
        if p.stat().st_size < int(min_bytes):
            return False, f"too_small<{min_bytes}B"
        dur = float(vp.ffprobe_duration_seconds(str(p)))
        if dur <= 0.05:
            return False, "zero_duration"
        return True, f"ok_dur={dur:.2f}s"
    except Exception as e:
        return False, f"error:{type(e).__name__}"


# ----------------------------
# Sidebar
# ----------------------------
with st.sidebar:
    st.header("🔑 OpenAI")
    st.session_state["api_key"] = st.text_input(
        "OpenAI API Key", type="password", value=st.session_state.get("api_key", "")
    ).strip()

    st.header("🎞 Output")
    st.session_state["mode"] = st.selectbox(
        "Mode",
        ["Long-form (16:9)", "Shorts / TikTok / Reels (9:16)"],
        index=0 if st.session_state.get("mode", "").startswith("Long") else 1,
    )
    if st.session_state["mode"].startswith("Long"):
        st.session_state["res_169"] = st.selectbox(
            "Resolution (16:9)",
            ["1280x720", "1920x1080"],
            index=0 if st.session_state.get("res_169") == "1280x720" else 1,
        )
    else:
        st.session_state["res_916"] = st.selectbox(
            "Resolution (9:16)",
            ["720x1280", "1080x1920"],
            index=1 if st.session_state.get("res_916") == "1080x1920" else 0,
        )

    st.header("☁️ Spaces Upload")
    auto_upload = st.toggle("Auto-upload to Spaces", value=True)
    make_public = st.toggle("Public-read ACL", value=True)
    prefix_override = st.text_input("Prefix override (optional)", value="").strip()

    st.divider()
    if st.button("🧹 Reset / Remove ZIP"):
        _reset_all()
        st.rerun()


# ----------------------------
# ZIP Upload
# ----------------------------
st.subheader("1) Upload ZIP (scripts + audio)")
uploaded = st.file_uploader("TTS Studio ZIP", type=["zip"])

if uploaded is not None:
    _reset_all()
    zip_root, zip_path = _save_uploaded_zip(uploaded)
    st.session_state["zip_root"] = zip_root
    st.session_state["zip_path"] = zip_path

    zip_bytes = Path(zip_path).read_bytes()
    workdir, extract_dir = vp.extract_zip_to_temp(zip_bytes)
    st.session_state["workdir"] = workdir
    st.session_state["extract_dir"] = extract_dir

    scripts, audios = vp.find_files(extract_dir)
    if not scripts:
        st.error("No script files (.txt/.md) found in ZIP.")
    elif not audios:
        st.error("No audio files found in ZIP.")
    else:
        pairs = vp.pair_segments(scripts, audios)

        missing = [Path(p["script_path"]).name for p in pairs if p.get("script_path") and not p.get("audio_path")]
        if missing:
            st.error("Missing audio for scripts: " + ", ".join(missing))
        else:
            st.session_state["pairs"] = pairs
            st.success(f"Detected {len(pairs)} segment(s).")

if st.session_state.get("pairs"):
    with st.expander("Detected segments"):
        for i, p in enumerate(st.session_state["pairs"], start=1):
            sp = Path(p.get("script_path") or "").name
            ap = Path(p.get("audio_path") or "").name
            st.write(f"{i:02d}. **{vp.segment_label(p)}** — {sp} ↔ {ap}")


# ----------------------------
# Generation
# ----------------------------
st.subheader("2) Generate MP4s")

pairs = st.session_state.get("pairs") or []
if not pairs:
    st.info("Upload a ZIP to enable generation.")
    st.stop()

w, h = _get_resolution_wh()
out_dir = _out_dir(st.session_state["extract_dir"])

c1, c2, c3 = st.columns([1, 1, 2])
with c1:
    overwrite = st.toggle("Overwrite existing", value=False)
with c2:
    if st.button("🛑 Stop"):
        st.session_state["stop_requested"] = True
        _log("🛑 Stop requested — will stop after current segment finishes.")
with c3:
    st.write(f"Output: **{w}x{h}** → `{out_dir}`")

go = st.button("🚀 Generate All", type="primary", disabled=st.session_state.get("is_generating", False))

if go:
    if not st.session_state.get("api_key"):
        st.error("OpenAI API key required for image generation.")
        st.stop()

    st.session_state["is_generating"] = True
    st.session_state["stop_requested"] = False
    st.session_state["generated"] = {}
    st.session_state["spaces_public_urls"] = []
    st.session_state["spaces_upload_log"] = []

    progress = st.progress(0.0)
    status = st.empty()
    detail = st.empty()

    # Spaces init
    s3 = bucket = region = public_base = None
    job_prefix = ""
    if auto_upload:
        try:
            s3, bucket, region, public_base = _spaces_client_and_context()
            job_prefix = prefix_override or _job_prefix()
            if not job_prefix.endswith("/"):
                job_prefix += "/"
            st.session_state["spaces_last_prefix"] = job_prefix
            _ulog(f"Auto-upload ON → bucket={bucket} region={region} prefix={job_prefix}")
        except Exception as e:
            auto_upload = False
            _ulog(f"❌ Auto-upload disabled: {type(e).__name__}: {e}")
            st.warning("Auto-upload disabled (Spaces init failed). See Upload Log.")

    n = len(pairs)
    for idx, pair in enumerate(pairs, start=1):
        if st.session_state.get("stop_requested"):
            _log("Stopped before next segment.")
            break

        out_path = _segment_out_path(out_dir, idx, pair)
        name = Path(out_path).name
        status.info(f"Generating {idx}/{n}: {name}")
        detail.caption(vp.segment_label(pair))

        if (not overwrite) and Path(out_path).exists():
            st.session_state["generated"][f"{idx:02d}"] = out_path
            _log(f"↪ skipped (exists): {name}")
        else:
            t0 = time.time()
            vp.render_segment_mp4(
                pair=pair,
                extract_dir=st.session_state["extract_dir"],
                out_path=out_path,
                api_key=st.session_state["api_key"],
                fps=30,
                width=w,
                height=h,
                max_scenes=vp.default_max_scenes(is_vertical=(h > w)),
                min_scene_seconds=6,
                max_scene_seconds=120,
                zoom_strength=0.0,
            )
            dt = time.time() - t0
            st.session_state["generated"][f"{idx:02d}"] = out_path
            _log(f"✅ {name} in {dt:.1f}s")

        # Upload
        if auto_upload and s3 and bucket and region:
            ok, why = _is_valid_mp4(out_path)
            if not ok:
                _ulog(f"⛔ skip upload (invalid {why}): {name}")
            else:
                try:
                    object_key = f"{job_prefix}{name}"
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
                    if url not in st.session_state["spaces_public_urls"]:
                        st.session_state["spaces_public_urls"].append(url)
                    _ulog(f"✅ {name} -> {url}")
                except Exception as e:
                    _ulog(f"❌ upload failed {name}: {type(e).__name__}: {e}")

        progress.progress(min(1.0, idx / max(1, n)))

    # Manifest
    if auto_upload and s3 and bucket and region:
        try:
            manifest = {
                "prefix": job_prefix,
                "mode": st.session_state.get("mode", ""),
                "resolution": f"{w}x{h}",
                "files": [Path(p).name for p in st.session_state["generated"].values()],
                "urls": st.session_state["spaces_public_urls"],
                "updated_at": datetime.now().isoformat(),
            }
            mkey = f"{job_prefix}manifest.json"
            murl = _upload_bytes_to_spaces(
                s3=s3,
                bucket=bucket,
                region=region,
                public_base=public_base or "",
                data=json.dumps(manifest, indent=2).encode("utf-8"),
                object_key=mkey,
                content_type="application/json",
                make_public=bool(make_public),
            )
            st.session_state["spaces_manifest_url"] = murl
            _ulog(f"📄 manifest.json -> {murl}")
        except Exception as e:
            _ulog(f"❌ manifest upload failed: {type(e).__name__}: {e}")

    st.session_state["is_generating"] = False
    status.success("Done.")


# ----------------------------
# Results + Logs
# ----------------------------
if st.session_state.get("generated"):
    st.subheader("3) Results (local)")
    for k, p in st.session_state["generated"].items():
        st.write(f"**{k}** — `{Path(p).name}`")
        st.video(p)

if st.session_state.get("spaces_public_urls"):
    st.subheader("4) Public URLs (Spaces)")
    for u in st.session_state["spaces_public_urls"]:
        st.write(u)

if st.session_state.get("spaces_manifest_url"):
    st.caption(f"Manifest: {st.session_state['spaces_manifest_url']}")

with st.expander("Generation Log"):
    for line in st.session_state.get("gen_log", []):
        st.write(line)

with st.expander("Upload Log"):
    for line in st.session_state.get("spaces_upload_log", []):
        st.write(line)
