# app_GODMODE.py
# UAPpress — Video Creator (GODMODE)
# ZIP (scripts+audio) → Images → Cached Clips → Concat → Mux Audio → (Optional Burn-in Subs) → MP4 → (Optional Spaces Upload)
#
# Rules:
# - Import-safe
# - Streamlit rerun-safe (all session_state keys pre-initialized)
# - Audio is authority
# - Ferrari UI: minimal + purposeful + fast

from __future__ import annotations

import os
import shutil
import tempfile
import zipfile
from dataclasses import asdict
from pathlib import Path
from typing import Tuple, List, Dict, Any

import streamlit as st

import video_pipeline as vp


# -----------------------------
# Streamlit config
# -----------------------------
st.set_page_config(page_title="UAPpress — Video Creator (GODMODE)", layout="wide")


# -----------------------------
# Session state init (RERUN SAFE)
# -----------------------------
def _ss_init() -> None:
    defaults = {
        # Core
        "openai_key": "",
        "mode": "Shorts / TikTok / Reels (9:16)",
        "resolution": "1080x1920",
        "fps": 24,
        "target_scene_sec": 6.0,
        "max_scenes": 30,
        # Subs
        "burn_subs": True,
        "subs_size": "Medium",
        "subs_safe_margin": 6,  # % of height
        # ZIP persistence
        "zip_saved_path": "",
        "extract_root": "",
        "detected_pairs": [],  # list[dict]
        # Spaces
        "spaces_enabled": False,
        "spaces_public": True,
        "spaces_prefix": "",
        "spaces_bucket": os.environ.get("DO_SPACES_BUCKET", ""),
        "spaces_region": os.environ.get("DO_SPACES_REGION", "nyc3"),
        "spaces_endpoint": os.environ.get("DO_SPACES_ENDPOINT", ""),  # optional override
        "spaces_key": os.environ.get("DO_SPACES_KEY", ""),
        "spaces_secret": os.environ.get("DO_SPACES_SECRET", ""),
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_ss_init()


# -----------------------------
# Helpers
# -----------------------------
def _vertical_mode(mode: str) -> bool:
    return "9:16" in (mode or "")


def _wh_from_resolution(res: str) -> Tuple[int, int]:
    try:
        w_s, h_s = (res or "").lower().split("x")
        return int(w_s.strip()), int(h_s.strip())
    except Exception:
        return 1080, 1920


def _subtitle_style_for_ui(w: int, h: int) -> Dict[str, Any]:
    size = st.session_state["subs_size"]
    if size == "Small":
        font_px = max(22, int(h * 0.030))
    elif size == "Large":
        font_px = max(28, int(h * 0.038))
    else:
        font_px = max(26, int(h * 0.034))

    margin_v = max(16, int(h * (st.session_state["subs_safe_margin"] / 100.0)))

    return {
        "font_name": "DejaVu Sans",
        "font_size": int(font_px),
        "outline": 2,
        "shadow": 1,
        "border_style": 3,  # boxed
        "alignment": 2,     # bottom-center
        "margin_v": int(margin_v),
    }


def _hard_reset_project() -> None:
    for key in ("zip_saved_path", "extract_root"):
        p = st.session_state.get(key, "")
        if p:
            try:
                pp = Path(p)
                if pp.exists():
                    if pp.is_file():
                        pp.unlink(missing_ok=True)
                    else:
                        shutil.rmtree(pp, ignore_errors=True)
            except Exception:
                pass
            st.session_state[key] = ""
    st.session_state["detected_pairs"] = []


def _save_uploaded_zip_to_tmp(uploaded_file) -> Path:
    root = Path(tempfile.mkdtemp(prefix="uappress_zip_"))
    zpath = root / "input.zip"
    zpath.write_bytes(uploaded_file.getbuffer())
    st.session_state["zip_saved_path"] = str(zpath)
    return zpath


def _extract_zip(zpath: Path) -> Path:
    root = Path(tempfile.mkdtemp(prefix="uappress_extract_"))
    extract_root = root / "extracted"
    extract_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(str(zpath), "r") as zf:
        zf.extractall(str(extract_root))
    st.session_state["extract_root"] = str(extract_root)
    return extract_root


def _detect_segments(extract_root: Path) -> List[vp.Segment]:
    scripts, audios = vp.find_files(extract_root)
    pairs = vp.pair_segments(scripts, audios)
    st.session_state["detected_pairs"] = [asdict(p) for p in pairs]
    return pairs


# -----------------------------
# Ferrari UI (minimal, functional)
# -----------------------------
st.markdown(
    """
    <style>
    .block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
    [data-testid="stSidebar"] { background: linear-gradient(180deg, #0b0f16 0%, #070a0f 100%); }
    [data-testid="stSidebar"] * { color: #EAF2FF; font-weight: 500; }
[data-testid="stSidebar"] input, [data-testid="stSidebar"] select { background-color: #101826; color: #EAF2FF; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🛸 UAPpress — Video Creator (GODMODE)")
st.caption("ZIP in → MP4 out → optional Spaces upload. Audio drives duration. Real caching under /tmp for speed.")

with st.sidebar:
    st.subheader("OpenAI")
    st.session_state["openai_key"] = st.text_input("OpenAI API Key", type="password", value=st.session_state["openai_key"])

    st.divider()
    st.subheader("Output")
    st.session_state["mode"] = st.selectbox("Mode", ["Shorts / TikTok / Reels (9:16)", "Long-form (16:9)"], index=0 if _vertical_mode(st.session_state["mode"]) else 1)

    if _vertical_mode(st.session_state["mode"]):
        res_options = ["1080x1920", "720x1280"]
        default_res = st.session_state["resolution"] if st.session_state["resolution"] in res_options else "1080x1920"
        st.session_state["resolution"] = st.selectbox("Resolution", res_options, index=res_options.index(default_res))
    else:
        res_options = ["1920x1080", "1280x720"]
        default_res = st.session_state["resolution"] if st.session_state["resolution"] in res_options else "1920x1080"
        st.session_state["resolution"] = st.selectbox("Resolution", res_options, index=res_options.index(default_res))

    st.session_state["fps"] = st.number_input("FPS", min_value=12, max_value=60, value=int(st.session_state["fps"]), step=1)

    st.divider()
    st.subheader("Scenes")
    st.session_state["target_scene_sec"] = st.slider(
        "Target seconds per scene",
        1.0, 12.0,
        float(st.session_state["target_scene_sec"]),
        0.5,
    )
    st.session_state["max_scenes"] = st.slider(
        "Max scenes per segment",
        2, 120,
        int(st.session_state["max_scenes"]),
        1,
    )
    st.session_state["image_budget"] = st.slider(
        "Image budget (token control)",
        2, 12,
        int(st.session_state.get("image_budget", 6)),
        1,
    )
    st.caption("If your cap would truncate audio, GODMODE overrides upward (audio is authority).")

    st.divider()
    st.subheader("Subtitles")
    st.session_state["burn_subs"] = st.checkbox("Burn-in subtitles (recommended)", value=bool(st.session_state["burn_subs"]))
    st.session_state["subs_size"] = st.selectbox("Subtitle size", ["Small", "Medium", "Large"], index=["Small", "Medium", "Large"].index(st.session_state["subs_size"]))
    st.session_state["subs_safe_margin"] = st.slider("Bottom safe margin (%)", 2, 14, int(st.session_state["subs_safe_margin"]), 1)

    st.divider()
    st.subheader("☁️ Spaces Upload (optional)")
    st.session_state["spaces_enabled"] = st.toggle("Auto-upload to Spaces", value=bool(st.session_state["spaces_enabled"]))
    st.session_state["spaces_public"] = st.toggle("Public-read ACL", value=bool(st.session_state["spaces_public"]))
    st.session_state["spaces_prefix"] = st.text_input("Prefix (optional)", value=st.session_state["spaces_prefix"])

    with st.expander("Spaces credentials", expanded=False):
        st.session_state["spaces_bucket"] = st.text_input("Bucket", value=st.session_state["spaces_bucket"])
        st.session_state["spaces_region"] = st.text_input("Region", value=st.session_state["spaces_region"])
        st.session_state["spaces_endpoint"] = st.text_input("Endpoint (optional)", value=st.session_state["spaces_endpoint"])
        st.session_state["spaces_key"] = st.text_input("Access Key", value=st.session_state["spaces_key"])
        st.session_state["spaces_secret"] = st.text_input("Secret Key", value=st.session_state["spaces_secret"], type="password")

    st.divider()
    st.caption(f"Cache: `{vp.cache_root()}`")
    if st.button("🧹 Reset / Remove ZIP", use_container_width=True):
        _hard_reset_project()
        st.rerun()


# -----------------------------
# Main flow
# -----------------------------
st.header("1) Upload ZIP (scripts + audio)")
uploaded = st.file_uploader("TTS Studio ZIP", type=["zip"])

if uploaded is not None:
    zpath = _save_uploaded_zip_to_tmp(uploaded)
    extract_root = _extract_zip(zpath)
    pairs = _detect_segments(extract_root)
    st.success(f"Detected {len(pairs)} segment(s).")

    with st.expander("Show detected segments", expanded=False):
        for i, seg in enumerate(pairs, 1):
            st.write(f"**{i}.** {seg.label}")
            st.write(f"- Script: `{seg.script_path}`")
            st.write(f"- Audio: `{seg.audio_path}`")
else:
    pairs = [vp.Segment(**d) for d in st.session_state.get("detected_pairs", [])] if st.session_state.get("detected_pairs") else []
    st.info("Upload/extract a ZIP to detect segments.")


st.header("2) Generate Segment MP4s")
if not pairs:
    st.warning("No segments detected yet.")
    st.stop()

w, h = _wh_from_resolution(st.session_state["resolution"])
vertical = _vertical_mode(st.session_state["mode"])
sub_style = _subtitle_style_for_ui(w, h)

gen_col1, gen_col2 = st.columns([1, 2], vertical_alignment="center")
with gen_col1:
    overwrite = st.checkbox("Overwrite existing MP4s", value=True)
with gen_col2:
    st.caption("Sequential render (crash-safe). Each MP4 validated before optional upload.")

if st.button("🎬 Generate MP4s", type="primary", use_container_width=True):
    if not st.session_state["openai_key"]:
        st.error("OpenAI API key is required.")
        st.stop()

    out_root = Path(tempfile.mkdtemp(prefix="uappress_out_"))
    out_dir = out_root / ("mp4_segments_9x16" if vertical else "mp4_segments_16x9")
    out_dir.mkdir(parents=True, exist_ok=True)
    st.write(f"Segments will be saved to: `{out_dir}`")

    spaces_cfg = None
    if st.session_state["spaces_enabled"]:
        spaces_cfg = vp.SpacesConfig(
            bucket=st.session_state["spaces_bucket"].strip(),
            region=st.session_state["spaces_region"].strip(),
            endpoint=st.session_state["spaces_endpoint"].strip() or None,
            access_key=st.session_state["spaces_key"].strip(),
            secret_key=st.session_state["spaces_secret"].strip(),
            public_read=bool(st.session_state["spaces_public"]),
            prefix=(st.session_state["spaces_prefix"] or "").strip(),
        )
        ok, msg = vp.validate_spaces_config(spaces_cfg)
        if not ok:
            st.error(f"Spaces config invalid: {msg}")
            st.stop()

    for idx, seg in enumerate(pairs, 1):
        st.subheader(f"Segment {idx}: {seg.label}")
        prog = st.progress(0.0)

        try:
            os.environ['UAPPRESS_IMAGE_BUDGET'] = str(st.session_state['image_budget'])
            mp4_path, meta = vp.render_segment_mp4(
                segment=seg,
                out_dir=out_dir,
                openai_api_key=st.session_state["openai_key"],
                width=w,
                height=h,
                fps=int(st.session_state["fps"]),
                vertical=vertical,
                target_scene_seconds=float(st.session_state["target_scene_sec"]),
                max_scenes=int(st.session_state["max_scenes"]),
                burn_subtitles=bool(st.session_state["burn_subs"]),
                subtitle_style=sub_style,
                overwrite=bool(overwrite),
                progress_cb=lambda p: prog.progress(min(1.0, max(0.0, float(p)))),
            )
        except Exception as e:
            st.error(f"Render failed: {e}")
            continue

        if meta.get("cap_overridden"):
            st.warning("Scene cap overridden upward to avoid truncating audio (audio is authority).")

        st.success(f"Rendered: `{mp4_path.name}`  (audio ~{meta.get('audio_duration', 0):.1f}s)")
        st.video(str(mp4_path))

        # Direct download (useful even with Spaces)
        st.download_button(
            "⬇️ Download MP4",
            data=mp4_path.read_bytes(),
            file_name=mp4_path.name,
            mime="video/mp4",
            use_container_width=True,
        )

        if spaces_cfg is not None:
            try:
                url = vp.upload_to_spaces(mp4_path, spaces_cfg)
                st.success(f"Uploaded to Spaces: {url}")
            except Exception as e:
                st.error(f"Upload failed: {e}")

    st.success("Done.")
