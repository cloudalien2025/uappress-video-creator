# app.py
# UAPpress Video Creator — GODMODE (Ferrari UI, Streamlit-safe)
# ZIP (scripts + audio + optional .srt) -> MP4 segments (audio-driven)
#
# Requirements:
# - Keep filenames: app.py, video_pipeline.py
# - ZIP workflow: non-negotiable
# - OpenAI used for images only (deterministic otherwise)
#
import hashlib
import tempfile
import zipfile
from pathlib import Path

import streamlit as st
import video_pipeline as vp

st.set_page_config(page_title="UAPpress — Video Creator (GODMODE)", layout="wide")

# ---------- helpers ----------
def _hash_bytes(b: bytes) -> str:
    return hashlib.sha1(b).hexdigest()[:12]

def _ensure_session_dirs() -> None:
    if "workdir" not in st.session_state:
        st.session_state.workdir = Path(tempfile.mkdtemp(prefix="uappress_vc_"))
    if "extract_dir" not in st.session_state:
        st.session_state.extract_dir = st.session_state.workdir / "extracted"
        st.session_state.extract_dir.mkdir(parents=True, exist_ok=True)
    if "out_dir" not in st.session_state:
        st.session_state.out_dir = st.session_state.workdir / "mp4_segments"
        st.session_state.out_dir.mkdir(parents=True, exist_ok=True)

def _reset_extract_dir() -> None:
    ed = st.session_state.extract_dir
    try:
        for p in ed.iterdir():
            if p.is_dir():
                vp._rmtree_safe(p)  # type: ignore
            else:
                p.unlink(missing_ok=True)  # py3.8 safe enough in cloud
    except Exception:
        pass
    ed.mkdir(parents=True, exist_ok=True)

_ensure_session_dirs()

# ---------- Ferrari skin ----------
st.markdown(
    """
<style>
/* Layout polish */
.block-container { padding-top: 1.1rem; padding-bottom: 2rem; }
h1, h2, h3 { letter-spacing: -0.02em; }
div[data-testid="stSidebar"] { background: #0b0e14; }
div[data-testid="stSidebar"] * { color: #e6e6e6; }
div[data-testid="stSidebar"] label { color: #cfd3da; }
div[data-testid="stSidebar"] .stSelectbox div { color: #111; }
.smallcaps { font-variant: all-small-caps; letter-spacing: .08em; opacity: .85; }
.card {
  border: 1px solid rgba(0,0,0,.06);
  border-radius: 18px;
  padding: 16px 18px;
  background: white;
}
.badge {
  display:inline-block; padding:4px 10px; border-radius:999px;
  border:1px solid rgba(0,0,0,.10); font-size:12px; margin-right:8px;
}
hr { border: none; border-top: 1px solid rgba(0,0,0,.06); margin: 18px 0; }
</style>
""",
    unsafe_allow_html=True,
)

# ---------- Header ----------
st.markdown('<div class="smallcaps">UAPpress</div>', unsafe_allow_html=True)
st.title("🛸 Video Creator — GODMODE")
st.caption("ZIP → Scenes → MP4 (audio-driven). Fast reruns via cache. Minimal knobs. Maximum throughput.")

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("## 🔑 API")
    openai_key = st.text_input("OpenAI API Key", type="password", help="Used for image generation only.")

    st.markdown("---")
    st.markdown("## 🎬 Output")
    mode = st.selectbox("Format", ["Shorts / TikTok / Reels (9:16)", "Long-form (16:9)"])
    is_vertical = "9:16" in mode

    if is_vertical:
        res = st.selectbox("Resolution", ["1080x1920", "720x1280"])
    else:
        res = st.selectbox("Resolution", ["1280x720", "1920x1080"])

    width, height = [int(x) for x in res.split("x")]

    fps = st.select_slider("FPS", options=[24, 25, 30], value=24)

    st.markdown("---")
    st.markdown("## 🧠 Scene Control")
    max_scenes = st.slider("Max scenes (images)", min_value=1, max_value=30, value=(6 if is_vertical else 12), step=1)
    min_scene_s = st.slider("Min seconds per scene", min_value=1, max_value=30, value=6, step=1)
    max_scene_s = st.slider("Max seconds per scene", min_value=5, max_value=180, value=120, step=5)

    # Clamp: never allow max < min (prevents 10s disasters)
    if max_scene_s < min_scene_s:
        max_scene_s = min_scene_s

    st.markdown("---")
    st.markdown("## 📝 Subtitles")
    burn_subs = st.checkbox("Burn-in subtitles (recommended)", value=True)
    subs_style = st.selectbox(
        "Subtitle style",
        ["Clean (default)", "Bold (high-contrast)"],
        index=0,
        help="Clean = documentary. Bold = louder for Shorts.",
        disabled=(not burn_subs),
    )

# ---------- Main: ZIP upload ----------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("1) Upload ZIP (scripts + audio + optional .srt)")
zip_file = st.file_uploader("TTS Studio ZIP", type=["zip"], label_visibility="collapsed")
st.markdown("</div>", unsafe_allow_html=True)

pairs = []
if zip_file is not None:
    zip_bytes = zip_file.read()
    zhash = _hash_bytes(zip_bytes)

    if st.session_state.get("zip_hash") != zhash:
        st.session_state.zip_hash = zhash
        _reset_extract_dir()
        zpath = st.session_state.workdir / f"input_{zhash}.zip"
        zpath.write_bytes(zip_bytes)
        with zipfile.ZipFile(zpath) as zf:
            zf.extractall(st.session_state.extract_dir)

    scripts, audios, subs = vp.find_files(st.session_state.extract_dir)
    pairs = vp.pair_segments(scripts, audios, subs)

    st.success(f"Detected {len(pairs)} segment(s).")
    with st.expander("Show detected segments"):
        for p in pairs:
            st.write(
                f"• **{vp.segment_label(p)}** — "
                f"{Path(p['audio_path']).name if p.get('audio_path') else 'NO AUDIO'} / "
                f"{Path(p['script_path']).name if p.get('script_path') else 'NO SCRIPT'} / "
                f"{Path(p['sub_path']).name if p.get('sub_path') else 'no .srt'}"
            )

# ---------- Generate ----------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("2) Generate MP4 segments")
colA, colB, colC = st.columns([1.4, 1, 1])

with colA:
    overwrite = st.checkbox("Overwrite existing MP4s", value=False)

with colB:
    st.write("")
    st.write("")
    go = st.button("🚀 Generate MP4s", use_container_width=True, disabled=(zip_file is None))

with colC:
    st.write("")
    st.write("")
    st.caption(f"Output folder: `{st.session_state.out_dir}`")

st.markdown("</div>", unsafe_allow_html=True)

if go:
    if not openai_key or not openai_key.strip():
        st.error("Enter your OpenAI API key in the sidebar.")
        st.stop()

    if not pairs:
        st.error("No segments found in ZIP.")
        st.stop()

    style = "clean" if subs_style.startswith("Clean") else "bold"

    prog = st.progress(0)
    status = st.empty()

    outputs = []
    for i, pair in enumerate(pairs, start=1):
        label = vp.segment_label(pair)
        out_name = f"{i:02d}_{vp.safe_slug(label + ' ' + (pair.get('title_guess') or ''))}_{pair.get('uid','')}.mp4"
        out_path = st.session_state.out_dir / out_name

        if out_path.exists() and (not overwrite):
            outputs.append(out_path)
            status.info(f"[{i}/{len(pairs)}] Cached: {out_path.name}")
            prog.progress(i / len(pairs))
            continue

        status.info(f"[{i}/{len(pairs)}] Rendering: {label}")
        try:
            vp.render_segment_mp4(
                pair=pair,
                extract_dir=str(st.session_state.extract_dir),
                out_path=str(out_path),
                api_key=openai_key.strip(),
                fps=int(fps),
                width=int(width),
                height=int(height),
                max_scenes=int(max_scenes),
                min_scene_seconds=int(min_scene_s),
                max_scene_seconds=int(max_scene_s),
                burn_subtitles=bool(burn_subs),
                subtitle_style=style,
            )
            outputs.append(out_path)
        except Exception as e:
            st.error(f"Failed on {label}: {e}")
            st.stop()

        prog.progress(i / len(pairs))

    st.success("Done.")
    st.divider()

    for p in outputs:
        st.video(str(p))
