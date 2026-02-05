# ============================
# app.py — UAPpress Video Creator
# PHOTOREALISTIC-ONLY • IMPORT-SAFE • STREAMLIT-CLOUD SAFE
# ============================

from __future__ import annotations

import os
import time
import zipfile
from pathlib import Path
from typing import List

import streamlit as st

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(
    page_title="UAPpress — Video Creator",
    layout="wide",
)

st.title("🛸 UAPpress — Video Creator")
st.caption("Photorealistic documentary video segments.")

# ----------------------------
# Sidebar — API + settings
# ----------------------------
with st.sidebar:
    st.header("🔑 API Settings")
    api_key = st.text_input("OpenAI API Key", type="password")

    st.header("🎥 Video Settings")
    max_scenes = st.slider("Max scenes", 3, 12, 6)
    min_scene_seconds = st.slider("Min scene seconds", 3, 10, 5)
    max_scene_seconds = st.slider("Max scene seconds", 6, 20, 10)
    zoom_strength = st.slider("Zoom strength", 0.0, 0.25, 0.08)

# ----------------------------
# Upload ZIP
# ----------------------------
st.header("1️⃣ Upload ZIP from TTS Studio (scripts + audio)")

zip_file = st.file_uploader(
    "Upload ZIP",
    type=["zip"],
    help="ZIP containing .txt scripts and matching .mp3 audio files",
)

if zip_file is None:
    st.stop()

workdir = Path("work")
workdir.mkdir(exist_ok=True)

zip_path = workdir / "input.zip"
zip_path.write_bytes(zip_file.read())

with zipfile.ZipFile(zip_path, "r") as z:
    z.extractall(workdir)

segments = sorted([p for p in workdir.glob("*.txt")])

if not segments:
    st.error("No .txt scripts found in ZIP")
    st.stop()

# ----------------------------
# Generate videos
# ----------------------------
st.header("2️⃣ Generate Segment MP4s")

outputs: List[str] = []

if st.button("🎬 Generate Segment MP4s"):
    if not api_key:
        st.error("Enter your OpenAI API key in the sidebar.")
        st.stop()

    os.environ["OPENAI_API_KEY"] = api_key

    # 🔒 SAFE IMPORT — avoids Streamlit boot crash
    try:
        from video_pipeline import render_segment_mp4
    except Exception as e:
        st.error("Failed to import video_pipeline.render_segment_mp4")
        st.exception(e)
        st.stop()

    for script_path in segments:
        seg_key = script_path.stem
        audio_path = script_path.with_suffix(".mp3")

        if not audio_path.exists():
            st.warning(f"Missing audio for {seg_key}, skipping.")
            continue

        st.info(f"Generating {seg_key}…")
        t0 = time.time()

        try:
            out_path = render_segment_mp4(
                script_path=str(script_path),
                audio_path=str(audio_path),
                max_scenes=int(max_scenes),
                min_scene_seconds=int(min_scene_seconds),
                max_scene_seconds=int(max_scene_seconds),
                zoom_strength=float(zoom_strength),
            )
        except Exception as e:
            st.error(f"Generation failed for {seg_key}")
            st.exception(e)
            continue

        outputs.append(out_path)
        st.success(f"✅ Generated {seg_key} in {time.time() - t0:.1f}s")

# ----------------------------
# Results
# ----------------------------
if outputs:
    st.header("📦 Outputs")
    for p in outputs:
        st.video(p)
