# app_GODMODE.txt
# Streamlit orchestration for fast, ZIP-driven video generation
# Design: thin UI shell, deterministic pipeline, rerun-safe, cost-aware

from __future__ import annotations
import io, os, zipfile, tempfile, shutil, time, json
from pathlib import Path
from typing import List, Dict, Tuple
import streamlit as st

import video_pipeline_GODMODE as vp

# ---------------- Page ----------------
st.set_page_config(page_title="GODMODE Video Factory", layout="wide")
st.title("⚡ GODMODE Video Factory")
st.caption("ZIP in → Viral MP4s out. Fast. Deterministic. Cheap.")

# ---------------- State ----------------
DEFAULTS = dict(
    api_key="",
    mode="Long-form (16:9)",
    res_169="1280x720",
    res_916="1080x1920",
    zip_root="",
    zip_path="",
    workdir="",
    extract_dir="",
    segments=[],
    generating=False,
    stop=False,
    outputs=[],
    log=[],
)
for k,v in DEFAULTS.items():
    st.session_state.setdefault(k, v)

def log(msg:str):
    st.session_state.log.append(msg)

# ---------------- Sidebar (Ferrari UI) ----------------
with st.sidebar:
    st.header("Engine")
    st.session_state.api_key = st.text_input("OpenAI API Key", type="password", value=st.session_state.api_key)
    st.session_state.mode = st.selectbox("Output", ["Long-form (16:9)", "Shorts (9:16)"])
    if st.session_state.mode.startswith("Long"):
        st.session_state.res_169 = st.selectbox("Resolution", ["1280x720","1920x1080"])
    else:
        st.session_state.res_916 = st.selectbox("Resolution", ["720x1280","1080x1920"])

# ---------------- ZIP Handling ----------------
st.subheader("1) Upload ZIP (scripts + audio)")
up = st.file_uploader("TTS Studio ZIP", type=["zip"])

def reset_zip():
    for k in ("zip_root","zip_path","workdir","extract_dir","segments","outputs"):
        if st.session_state.get(k):
            try:
                shutil.rmtree(st.session_state[k], ignore_errors=True)
            except Exception:
                pass
            st.session_state[k] = "" if isinstance(st.session_state[k], str) else []
    st.session_state.log = []

if st.button("Reset"):
    reset_zip()
    st.experimental_rerun()

if up is not None:
    reset_zip()
    root = tempfile.mkdtemp(prefix="god_zip_")
    zp = os.path.join(root,"upload.zip")
    with open(zp,"wb") as f: f.write(up.getbuffer())
    st.session_state.zip_root, st.session_state.zip_path = root, zp
    with open(zp,"rb") as f: data = f.read()
    workdir, extract_dir = vp.extract_zip_to_temp(data)
    st.session_state.workdir, st.session_state.extract_dir = workdir, extract_dir
    scripts, audios = vp.find_files(extract_dir)
    if not scripts or not audios:
        st.error("ZIP must contain scripts and audio.")
    else:
        pairs = vp.pair_segments(scripts, audios)
        st.session_state.segments = pairs
        st.success(f"Detected {len(pairs)} segments")

# ---------------- Generate ----------------
st.subheader("2) Generate")
if st.session_state.segments:
    if st.button("🚀 Generate All"):
        st.session_state.generating = True
        st.session_state.stop = False
        st.session_state.outputs = []
        w,h = vp.resolution_wh(st.session_state.mode, st.session_state.res_169, st.session_state.res_916)
        outdir = Path(st.session_state.extract_dir)/("out_916" if h>w else "out_169")
        outdir.mkdir(exist_ok=True)
        for i,pair in enumerate(st.session_state.segments, start=1):
            if st.session_state.stop: break
            out = outdir/f"{i:02d}_{vp.safe_slug(vp.segment_label(pair))}.mp4"
            t0=time.time()
            vp.render_segment_mp4(
                pair=pair,
                extract_dir=st.session_state.extract_dir,
                out_path=str(out),
                api_key=st.session_state.api_key,
                width=w,height=h,fps=30,
                max_scenes=vp.default_scenes(h>w),
            )
            st.session_state.outputs.append(str(out))
            log(f"✓ {out.name} in {time.time()-t0:.1f}s")
        st.session_state.generating=False

if st.session_state.outputs:
    st.success("Done")
    for o in st.session_state.outputs:
        st.video(o)

with st.expander("Log"):
    for l in st.session_state.log:
        st.write(l)
