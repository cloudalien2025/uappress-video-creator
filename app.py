# app.py
"""
UAPpress — Video Creator (GODMODE + Ferrari UI)

ZIP in (MP3 + scripts) -> deterministic pairing -> images -> video -> mux audio -> (optional) burn-in subtitles -> MP4
Optional: Auto-upload finished MP4s to DigitalOcean Spaces (S3-compatible)

Streamlit Cloud realities:
- reruns can happen at any line
- ephemeral filesystem
- no background workers
- import-time must be safe
"""

from __future__ import annotations

import json
import os
import time
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

# Video engine (same repo file: video_pipeline.py)
try:
    import video_pipeline as vp
except Exception as e:
    st.error(f"Failed to import video_pipeline.py: {type(e).__name__}: {e}")
    st.stop()

# Spaces (S3)
import boto3
from botocore.client import Config
from botocore.exceptions import ClientError


# ----------------------------
# Import-time safe helpers
# ----------------------------

def _ss_init() -> None:
    """Initialize all session_state keys used anywhere (rerun-safe)."""
    defaults: Dict[str, Any] = {
        "api_key": "",
        "mode": "Shorts / TikTok / Reels (9:16)",
        "res_916": "1080x1920",
        "res_169": "1920x1080",
        "fps": 24,
        "burn_subs": True,
        "subs_size": "Medium",
        "subs_safe_margin": 6,  # % of height
        "overwrite": False,

        # scene timing knobs
        "target_scene_sec": 4.0,  # used to compute scene count
                "max_scenes": 14,

        # zoom
        "zoom_strength": 0.0,

        # ZIP state
        "workdir": "",
        "extract_dir": "",
        "pairs": [],
        "generated": {},

        # logs
        "gen_log": [],
        "spaces_upload_log": [],
        "spaces_public_urls": [],
        "spaces_manifest_url": "",

        # controls
        "stop_requested": False,
        "is_generating": False,

        # Spaces toggles
        "auto_upload": True,
        "make_public": True,
        "prefix_override": "",
        "spaces_last_prefix": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _log(msg: str) -> None:
    st.session_state["gen_log"].append(msg)


def _ulog(msg: str) -> None:
    st.session_state["spaces_upload_log"].append(msg)


def _reset_job_state(keep_api_key: bool = True) -> None:
    api_key = st.session_state.get("api_key", "")
    _ss_init()
    if keep_api_key:
        st.session_state["api_key"] = api_key
    # clean up temp workdir if exists
    try:
        wd = st.session_state.get("workdir") or ""
        if wd and Path(wd).exists():
            vp.safe_rmtree(wd)
    except Exception:
        pass
    st.session_state["workdir"] = ""
    st.session_state["extract_dir"] = ""
    st.session_state["pairs"] = []
    st.session_state["generated"] = {}
    st.session_state["spaces_public_urls"] = []
    st.session_state["spaces_manifest_url"] = ""
    st.session_state["gen_log"] = []
    st.session_state["spaces_upload_log"] = []
    st.session_state["stop_requested"] = False
    st.session_state["is_generating"] = False


def _get_resolution_wh() -> Tuple[int, int]:
    if st.session_state["mode"].startswith("Long"):
        w, h = st.session_state["res_169"].split("x")
    else:
        w, h = st.session_state["res_916"].split("x")
    return int(w), int(h)


def _job_prefix() -> str:
    return datetime.now().strftime("uappress/%Y-%m-%d/%H%M%S")


def _spaces_client_and_context() -> Tuple[Any, str, str, str]:
    """
    Reads standard DO Spaces env vars:
      - DO_SPACES_KEY, DO_SPACES_SECRET
      - DO_SPACES_REGION (e.g. nyc3)
      - DO_SPACES_BUCKET
      - DO_SPACES_ENDPOINT (optional; default https://{region}.digitaloceanspaces.com)
    """
    key = os.environ.get("DO_SPACES_KEY", "").strip()
    secret = os.environ.get("DO_SPACES_SECRET", "").strip()
    region = os.environ.get("DO_SPACES_REGION", "").strip() or "nyc3"
    bucket = os.environ.get("DO_SPACES_BUCKET", "").strip()
    endpoint = os.environ.get("DO_SPACES_ENDPOINT", "").strip() or f"https://{region}.digitaloceanspaces.com"

    if not key or not secret or not bucket:
        raise RuntimeError("Missing DO Spaces env vars. Need DO_SPACES_KEY, DO_SPACES_SECRET, DO_SPACES_BUCKET.")

    s3 = boto3.client(
        "s3",
        region_name=region,
        endpoint_url=endpoint,
        aws_access_key_id=key,
        aws_secret_access_key=secret,
        config=Config(signature_version="s3v4"),
    )
    public_base = endpoint.rstrip("/") + f"/{bucket}"
    return s3, bucket, region, public_base


def _upload_file_to_spaces(
    *,
    s3: Any,
    bucket: str,
    region: str,
    public_base: str,
    local_path: str,
    object_key: str,
    make_public: bool,
    skip_if_exists: bool = True,
) -> str:
    p = Path(local_path)
    if not p.exists():
        raise RuntimeError(f"Local file missing: {p}")
    if skip_if_exists:
        try:
            s3.head_object(Bucket=bucket, Key=object_key)
            # already exists
            return f"{public_base}/{object_key}"
        except Exception:
            pass

    extra_args: Dict[str, Any] = {"ContentType": "video/mp4"}
    if make_public:
        extra_args["ACL"] = "public-read"

    s3.upload_file(str(p), bucket, object_key, ExtraArgs=extra_args)
    return f"{public_base}/{object_key}"


def _upload_bytes_to_spaces(
    *,
    s3: Any,
    bucket: str,
    region: str,
    public_base: str,
    data: bytes,
    object_key: str,
    content_type: str,
    make_public: bool,
) -> str:
    extra_args: Dict[str, Any] = {"ContentType": content_type}
    if make_public:
        extra_args["ACL"] = "public-read"
    s3.put_object(Bucket=bucket, Key=object_key, Body=data, **extra_args)
    return f"{public_base}/{object_key}"


def _is_valid_mp4(path: str) -> Tuple[bool, str]:
    return vp.validate_mp4(path)


def _out_dir(extract_dir: str, w: int, h: int) -> str:
    base = Path(extract_dir) / "_mp4_segments"
    base = base / (f"{w}x{h}")
    base.mkdir(parents=True, exist_ok=True)
    return str(base)


def _segment_out_path(out_dir: str, idx: int, pair: Dict[str, Any]) -> str:
    label = vp.safe_slug(vp.segment_label(pair), 18)
    stem = Path(pair.get("audio_path") or pair.get("script_path") or f"seg_{idx:02d}").stem
    stem = vp.safe_slug(stem, 40)
    return str(Path(out_dir) / f"{idx:02d}_{label}_{stem}.mp4")


def _subtitle_style_for_ui(w: int, h: int) -> Dict[str, Any]:
    size = st.session_state["subs_size"]
    # scale font size with height; tuned for Shorts readability without clown letters
    if size == "Small":
        font_px = max(24, int(h * 0.030))
    elif size == "Large":
        font_px = max(30, int(h * 0.038))
    else:
        font_px = max(28, int(h * 0.034))

    margin_v = max(16, int(h * (st.session_state["subs_safe_margin"] / 100.0)))
    # Alignment=2 (bottom-center)
    return {
        "font_name": "DejaVu Sans",
        "font_size": int(font_px),
        "outline": 2,
        "shadow": 1,
        "border_style": 3,  # boxed
        "alignment": 2,
        "margin_v": int(margin_v),
    }


# ----------------------------
# Page + Ferrari UI
# ----------------------------
_ss_init()

st.set_page_config(page_title="UAPpress — Video Creator", layout="wide")

st.markdown(
    """
    <style>
      :root { --bg:#0b0e14; --panel:#111827; --card:#ffffff; --muted: rgba(49,51,63,.68); }
      .block-container { padding-top: 1.2rem; padding-bottom: 2.2rem; max-width: 1200px; }
      h1,h2,h3 { letter-spacing: -0.02em; }
      .small-muted { color: var(--muted); font-size: 0.95rem; }
      .card { border:1px solid rgba(0,0,0,.06); border-radius: 18px; padding: 16px 18px; background: var(--card); }
      .card + .card { margin-top: 14px; }
      /* Sidebar dark */
      section[data-testid='stSidebar'] { background: var(--bg); }
      section[data-testid='stSidebar'] * { color: #e6e6e6; }
      section[data-testid='stSidebar'] input, section[data-testid='stSidebar'] textarea { color: #111; }
      section[data-testid='stSidebar'] .stSelectbox div[data-baseweb='select'] { color:#111; }
      /* Buttons */
      div.stButton > button { border-radius: 14px; padding: .65rem .95rem; }
      /* Expanders */
      div[data-testid='stExpander'] { border-radius: 14px; }
      /* Video */
      video { border-radius: 14px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("<div class='small-muted'>UAPpress</div>", unsafe_allow_html=True)
st.title("🛸 Video Creator — GODMODE")
st.markdown(
    "<div class='small-muted'>ZIP in → MP4 out → optional Spaces upload. Audio drives duration. Burn‑in subtitles default ON. Speed wins.</div>",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("🔑 OpenAI")
    st.session_state["api_key"] = st.text_input("OpenAI API Key", type="password", value=st.session_state["api_key"]).strip()

    st.header("🎞 Output")
    st.session_state["mode"] = st.selectbox(
        "Mode",
        ["Shorts / TikTok / Reels (9:16)", "Long-form (16:9)"],
        index=0 if st.session_state["mode"].startswith("Short") else 1,
    )

    if st.session_state["mode"].startswith("Long"):
        st.session_state["res_169"] = st.selectbox(
            "Resolution (16:9)",
            ["1280x720", "1920x1080"],
            index=1 if st.session_state["res_169"] == "1920x1080" else 0,
        )
    else:
        st.session_state["res_916"] = st.selectbox(
            "Resolution (9:16)",
            ["720x1280", "1080x1920"],
            index=1 if st.session_state["res_916"] == "1080x1920" else 0,
        )

    st.session_state["fps"] = st.number_input("FPS", min_value=15, max_value=60, value=int(st.session_state["fps"]), step=1)

    st.header("🧩 Scenes")
    st.session_state["target_scene_sec"] = st.slider(
        "Target seconds per scene (auto scene count)",
        min_value=1.0,
        max_value=12.0,
        value=float(st.session_state["target_scene_sec"]),
        step=0.5,
    )
    st.session_state["max_scenes"] = st.slider("Max scenes per segment", 2, 80, int(st.session_state["max_scenes"]), 1)

    st.caption("Note: GODMODE will *always* cover full audio duration. If your caps would truncate audio, the engine auto-increases scene count (and logs it).")

    st.header("🧷 Subtitles")

    st.session_state["burn_subs"] = st.checkbox("Burn‑in subtitles (recommended)", value=bool(st.session_state["burn_subs"]))
    st.session_state["subs_size"] = st.selectbox("Subtitle size", ["Small", "Medium", "Large"], index=["Small", "Medium", "Large"].index(st.session_state["subs_size"]))
    st.session_state["subs_safe_margin"] = st.slider("Safe bottom margin (%)", 2, 14, int(st.session_state["subs_safe_margin"]), 1)

    st.header("☁️ Spaces Upload")
    st.session_state["auto_upload"] = st.toggle("Auto-upload to Spaces", value=bool(st.session_state["auto_upload"]))
    st.session_state["make_public"] = st.toggle("Public-read ACL", value=bool(st.session_state["make_public"]))
    st.session_state["prefix_override"] = st.text_input("Prefix override (optional)", value=st.session_state["prefix_override"]).strip()

    st.divider()
    if st.button("🧹 Reset / Remove ZIP"):
        _reset_job_state(keep_api_key=True)
        st.rerun()


# ----------------------------
# ZIP Upload
# ----------------------------
st.subheader("1) Upload ZIP (scripts + audio)")
uploaded = st.file_uploader("TTS Studio ZIP", type=["zip"])

if uploaded is not None:
    # new upload resets job state, but keeps api key
    _reset_job_state(keep_api_key=True)

    zip_bytes = uploaded.read()
    workdir, extract_dir = vp.extract_zip_to_temp(zip_bytes)
    st.session_state["workdir"] = workdir
    st.session_state["extract_dir"] = extract_dir

    scripts, audios = vp.find_files(extract_dir)
    if not scripts:
        st.error("No script files (.txt/.md) found in ZIP.")
    elif not audios:
        st.error("No audio files (.mp3/.wav/.m4a/...) found in ZIP.")
    else:
        pairs = vp.pair_segments(scripts, audios)
        st.session_state["pairs"] = pairs
        st.success(f"Detected {len(pairs)} segment(s).")

pairs: List[Dict[str, Any]] = st.session_state.get("pairs") or []
if pairs:
    with st.expander("Show detected segments"):
        for i, p in enumerate(pairs, start=1):
            sp = Path(p.get("script_path") or "").name
            ap = Path(p.get("audio_path") or "").name
            st.write(f"{i:02d}. **{vp.segment_label(p)}** — {sp} ↔ {ap}")

st.subheader("2) Generate Segment MP4s")

if not pairs:
    st.info("Upload a ZIP to enable generation.")
    st.stop()

w, h = _get_resolution_wh()
out_dir = _out_dir(st.session_state["extract_dir"], w, h)

c1, c2, c3 = st.columns([1, 1, 2])
with c1:
    st.session_state["overwrite"] = st.toggle("Overwrite existing MP4s", value=bool(st.session_state["overwrite"]))
with c2:
    if st.button("🛑 Stop"):
        st.session_state["stop_requested"] = True
        _log("🛑 Stop requested — will stop after current segment finishes.")
with c3:
    st.write(f"Output: **{w}x{h}** → `{out_dir}`")

go = st.button("🚀 Generate All", type="primary", disabled=bool(st.session_state.get("is_generating", False)))

if go:
    if not st.session_state["api_key"]:
        st.error("OpenAI API key required for image generation.")
        st.stop()

    st.session_state["is_generating"] = True
    st.session_state["stop_requested"] = False
    st.session_state["generated"] = {}
    st.session_state["spaces_public_urls"] = []
    st.session_state["spaces_manifest_url"] = ""
    st.session_state["gen_log"] = []
    st.session_state["spaces_upload_log"] = []

    progress = st.progress(0.0)
    status = st.empty()
    detail = st.empty()

    # Spaces init (optional)
    s3 = bucket = region = public_base = None
    job_prefix = ""
    if st.session_state["auto_upload"]:
        try:
            s3, bucket, region, public_base = _spaces_client_and_context()
            job_prefix = st.session_state["prefix_override"] or _job_prefix()
            if job_prefix and not job_prefix.endswith("/"):
                job_prefix += "/"
            st.session_state["spaces_last_prefix"] = job_prefix
            _ulog(f"Auto-upload ON → bucket={bucket} region={region} prefix={job_prefix}")
        except Exception as e:
            _ulog(f"❌ Auto-upload disabled: {type(e).__name__}: {e}")
            st.warning("Auto-upload disabled (Spaces init failed). See Upload Log.")
            s3 = bucket = region = public_base = None

    n = len(pairs)
    for idx, pair in enumerate(pairs, start=1):
        if st.session_state.get("stop_requested"):
            _log("Stopped before next segment.")
            break

        out_path = _segment_out_path(out_dir, idx, pair)
        name = Path(out_path).name
        status.info(f"Generating {idx}/{n}: {name}")
        detail.caption(vp.segment_label(pair))

        if (not st.session_state["overwrite"]) and Path(out_path).exists():
            st.session_state["generated"][f"{idx:02d}"] = out_path
            _log(f"↪ skipped (exists): {name}")
        else:
            t0 = time.time()

            sub_style = _subtitle_style_for_ui(w, h)
            vp.render_segment_mp4(
                pair=pair,
                extract_dir=st.session_state["extract_dir"],
                out_path=out_path,
                api_key=st.session_state["api_key"],
                fps=int(st.session_state["fps"]),
                width=w,
                height=h,
                max_scenes=int(st.session_state["max_scenes"]),
                target_scene_seconds=float(st.session_state["target_scene_sec"]),
                                zoom_strength=float(st.session_state["zoom_strength"]),
                burn_subtitles=bool(st.session_state["burn_subs"]),
                subtitle_style=sub_style,
            )

            dt = time.time() - t0
            st.session_state["generated"][f"{idx:02d}"] = out_path
            _log(f"✅ {name} in {dt:.1f}s")

        # Upload gated by validation
        if s3 and bucket and region and public_base:
            ok, why = _is_valid_mp4(out_path)
            if not ok:
                _ulog(f"⛔ skip upload (invalid {why}): {name}")
            else:
                try:
                    object_key = f"{job_prefix}{name}" if job_prefix else name
                    url = _upload_file_to_spaces(
                        s3=s3,
                        bucket=bucket,
                        region=region,
                        public_base=public_base,
                        local_path=out_path,
                        object_key=object_key,
                        make_public=bool(st.session_state["make_public"]),
                        skip_if_exists=True,
                    )
                    st.session_state["spaces_public_urls"].append(url)
                    _ulog(f"✅ {name} -> {url}")
                except Exception as e:
                    _ulog(f"❌ upload failed {name}: {type(e).__name__}: {e}")

        progress.progress(min(1.0, idx / max(1, n)))

    # Manifest (optional)
    if s3 and bucket and region and public_base and st.session_state["spaces_public_urls"]:
        try:
            manifest = {
                "prefix": job_prefix,
                "mode": st.session_state.get("mode", ""),
                "resolution": f"{w}x{h}",
                "files": [Path(p).name for p in st.session_state["generated"].values()],
                "urls": st.session_state["spaces_public_urls"],
                "updated_at": datetime.now().isoformat(),
            }
            mkey = f"{job_prefix}manifest.json" if job_prefix else "manifest.json"
            murl = _upload_bytes_to_spaces(
                s3=s3,
                bucket=bucket,
                region=region,
                public_base=public_base,
                data=json.dumps(manifest, indent=2).encode("utf-8"),
                object_key=mkey,
                content_type="application/json",
                make_public=bool(st.session_state["make_public"]),
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
