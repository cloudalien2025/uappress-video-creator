# app.py — UAPpress Video Creator (Zip-in → MP4-out)
import os
import io
import zipfile
import tempfile
import asyncio
from typing import List, Dict

import streamlit as st
from openai import OpenAI, AsyncOpenAI

from video_pipeline import (
    find_files,
    best_match_pairs,
    read_script_file,
    plan_scenes,
    generate_clip_bytes,
    write_bytes,
    build_chapter_mp4,
    build_full_documentary,
    safe_slug,
    reencode_if_needed,
)

st.set_page_config(page_title="UAPpress Video Creator", layout="wide")
st.title("🎬 UAPpress — Zip → Documentary MP4")
st.caption("Upload the ZIP exported by Documentary TTS Studio (scripts + MP3s). Generate clips per chapter and stitch into one MP4.")

with st.sidebar:
    st.header("🔑 OpenAI Settings")
    api_key = st.text_input("OpenAI API Key", type="password")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

    st.divider()
    st.header("🎥 Video Settings")
    video_model = st.selectbox("Video model", ["sora-2", "sora-2-pro"], index=0)
    size = st.selectbox("Resolution", ["1280x720", "720x1280"], index=0)
    seconds = st.selectbox("Seconds per scene", [4, 8, 12], index=1)
    max_scenes = st.slider("Max scenes per chapter", 3, 14, 8)

    st.divider()
    st.header("🧠 Scene Planner")
    text_model = st.selectbox("Text model", ["gpt-5-mini", "gpt-5"], index=0)

    st.divider()
    st.header("⚙️ Run Mode")
    dry_run = st.checkbox("Dry run (generate scene plan only)", value=True)
    only_first_chapter = st.checkbox("Only render first chapter", value=True)
    force_reencode = st.checkbox("Force re-encode chapter video (slower, more compatible)", value=False)

st.subheader("1) Upload Documentary ZIP")
zip_file = st.file_uploader("Upload ZIP from Documentary TTS Studio", type=["zip"])

if "pairs" not in st.session_state:
    st.session_state.pairs = []
if "workdir" not in st.session_state:
    st.session_state.workdir = None
if "outputs" not in st.session_state:
    st.session_state.outputs = {}

if zip_file:
    # Extract into a persistent temp dir for this session
    if st.session_state.workdir is None:
        st.session_state.workdir = tempfile.mkdtemp(prefix="uappress_video_")

    workdir = st.session_state.workdir
    extract_dir = os.path.join(workdir, "extracted")
    os.makedirs(extract_dir, exist_ok=True)

    # clean previous extraction
    for root, dirs, files in os.walk(extract_dir):
        for f in files:
            try:
                os.remove(os.path.join(root, f))
            except Exception:
                pass

    with zipfile.ZipFile(zip_file, "r") as z:
        z.extractall(extract_dir)

    scripts, audios = find_files(extract_dir)
    pairs = best_match_pairs(scripts, audios)
    st.session_state.pairs = pairs

    st.success(f"Extracted ZIP. Found {len(scripts)} script file(s) and {len(audios)} audio file(s).")

st.subheader("2) Review Chapter Pairing")
pairs: List[Dict] = st.session_state.pairs

if pairs:
    colA, colB = st.columns([2, 2])
    with colA:
        st.write("**Detected chapters (script ↔ audio):**")
    with colB:
        st.write("**Tip:** If a pairing looks wrong, rename files in the source ZIP with chapter numbers like `01`, `02`, etc.")

    for i, p in enumerate(pairs, start=1):
        st.write(f"**{i}. {p['title_guess']}**")
        st.code(f"SCRIPT: {p['script_path']}\nAUDIO:  {p['audio_path']}", language="text")

else:
    st.info("Upload a ZIP to detect chapters.")
    st.stop()

st.subheader("3) Generate Video")

if st.button("🚀 Build MP4"):
    if not api_key:
        st.error("Enter your OpenAI API key in the sidebar.")
        st.stop()

    sync_client = OpenAI()
    async_client = AsyncOpenAI()

    workdir = st.session_state.workdir or tempfile.mkdtemp(prefix="uappress_video_")
    out_dir = os.path.join(workdir, "render_out")
    os.makedirs(out_dir, exist_ok=True)

    to_run = pairs[:1] if only_first_chapter else pairs

    chapter_mp4s = []
    progress = st.progress(0.0)
    status = st.empty()

    total_scene_budget = len(to_run) * max_scenes
    done = 0

    for idx, p in enumerate(to_run, start=1):
        title = p["title_guess"]
        chapter_slug = f"chapter_{idx:02d}_{safe_slug(title)}"
        chapter_dir = os.path.join(out_dir, chapter_slug)
        os.makedirs(chapter_dir, exist_ok=True)

        chapter_text = read_script_file(p["script_path"])
        mp3_path = p["audio_path"]

        status.write(f"### Chapter {idx}: {title}")

        # 1) plan scenes
        scenes = plan_scenes(
            sync_client,
            chapter_title=title,
            chapter_text=chapter_text,
            max_scenes=max_scenes,
            seconds_per_scene=int(seconds),
            model=text_model,
        )

        st.write("**Scene plan:**")
        st.json(scenes)

        if dry_run:
            # save scene plan for download
            plan_path = os.path.join(chapter_dir, "scene_plan.json")
            with open(plan_path, "w", encoding="utf-8") as f:
                f.write(__import__("json").dumps(scenes, ensure_ascii=False, indent=2))
            st.success("Dry run complete for this chapter (scene plan only).")
            continue

        # 2) generate clips
        scene_paths = []
        for sc in scenes:
            sc_no = sc["scene"]
            sc_seconds = int(sc.get("seconds", seconds))
            prompt = sc["prompt"]

            status.write(f"Generating Chapter {idx} — Scene {sc_no} ({sc_seconds}s)")
            clip_path = os.path.join(chapter_dir, f"scene_{sc_no:02d}.mp4")

            # cache: don’t regenerate if already exists
            if not os.path.exists(clip_path):
                clip_bytes = asyncio.run(
                    generate_clip_bytes(
                        async_client,
                        prompt=prompt,
                        seconds=sc_seconds,
                        size=size,
                        model=video_model,
                    )
                )
                write_bytes(clip_path, clip_bytes)

            scene_paths.append(clip_path)

            done += 1
            progress.progress(min(done / max(total_scene_budget, 1), 1.0))

        # 3) stitch + mux audio
        status.write(f"Stitching Chapter {idx} video + audio…")
        chapter_mp4 = build_chapter_mp4(scene_paths, mp3_path, chapter_dir, chapter_slug)

        # Optional: force re-encode for compatibility
        if force_reencode:
            reencoded = os.path.join(chapter_dir, f"{chapter_slug}_final_reencoded.mp4")
            chapter_mp4 = reencode_if_needed(chapter_mp4, reencoded)

        chapter_mp4s.append(chapter_mp4)

        st.success(f"Chapter MP4 ready: {os.path.basename(chapter_mp4)}")
        with open(chapter_mp4, "rb") as f:
            st.download_button(
                f"Download Chapter {idx} MP4",
                data=f,
                file_name=os.path.basename(chapter_mp4),
                mime="video/mp4",
                key=f"dl_ch_{idx}",
            )

    # 4) full concat (only if not dry run and >1)
    if (not dry_run) and (not only_first_chapter) and len(chapter_mp4s) > 1:
        status.write("Concatenating full documentary…")
        final_path = os.path.join(out_dir, "uappress_documentary_final.mp4")
        build_full_documentary(chapter_mp4s, final_path)

        if force_reencode:
            reencoded = os.path.join(out_dir, "uappress_documentary_final_reencoded.mp4")
            final_path = reencode_if_needed(final_path, reencoded)

        st.success("Full documentary MP4 ready!")
        with open(final_path, "rb") as f:
            st.download_button(
                "Download Full Documentary MP4",
                data=f,
                file_name=os.path.basename(final_path),
                mime="video/mp4",
            )

    status.write("✅ Done.")
